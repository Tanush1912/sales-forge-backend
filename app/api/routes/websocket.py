"""WebSocket handler for real-time voice calls using Gemini Live API."""

import asyncio
import base64
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, Query

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.db import get_session_factory
from app.models.database import (
    Session,
    Scenario,
    Transcript,
    Feedback,
    SessionAnalytics,
    ConversationState,
    User,
)
from app.models.schemas import WSMessageType, WSServerMessage
from app.core.llm.orchestrator import ConversationOrchestrator
from app.core.llm.prompt_builder import PersonaContext, ScenarioContext
from app.core.voice.live_api_client import LiveAPIClient, LiveSessionConfig
from app.api.routes.auth import validate_jwt
from app.config import get_settings

DEMO_EMAIL = "demo@salesforge.app"
DEMO_DAILY_SESSION_LIMIT = 10

logger = logging.getLogger(__name__)

router = APIRouter()


def build_persona_context(scenario: Scenario) -> PersonaContext:
    """Build PersonaContext from scenario's persona."""
    persona = scenario.persona
    return PersonaContext(
        name=persona.name if persona else "Unknown",
        title=persona.title if persona else "Unknown",
        company=persona.company if persona else "Unknown",
        industry=persona.industry if persona else None,
        traits=persona.traits if persona else [],
        default_mood=persona.default_mood if persona else "neutral",
        system_prompt_template=persona.system_prompt_template if persona else "",
        behavior_config=persona.behavior_config if persona else {},
    )


def build_scenario_context(scenario: Scenario) -> ScenarioContext:
    """Build ScenarioContext from scenario."""
    return ScenarioContext(
        type=scenario.type,
        category=scenario.category,
        instructions=scenario.instructions,
        scenario_rules=scenario.scenario_rules,
        success_criteria=scenario.success_criteria or {},
        prior_context=scenario.prior_context,
    )


async def send_message(websocket: WebSocket, message: WSServerMessage):
    """Send a message to the client."""
    await websocket.send_json(message.model_dump())


@router.websocket("/{session_id}")
async def call_websocket(
    websocket: WebSocket,
    session_id: UUID,
    token: str | None = Query(None),
):
    """WebSocket endpoint for real-time voice calls using Gemini Live API.

    Message flow:
    1. Client connects with session_id
    2. Server validates session and connects to Gemini Live
    3. Client streams audio chunks → forwarded to Gemini Live
    4. Gemini Live streams audio response → forwarded to client
    5. State updates sent periodically via context injection
    6. Client sends end_call to terminate
    7. Server generates feedback and saves results
    """
    
    if not token:
        await websocket.accept()
        await send_message(
            websocket,
            WSServerMessage(type=WSMessageType.ERROR, error="Authentication token missing"),
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        jwt_payload = await validate_jwt(token)
    except Exception as e:
        await websocket.accept()
        await send_message(
            websocket,
            WSServerMessage(type=WSMessageType.ERROR, error=f"Authentication failed: {str(e)}"),
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    factory = get_session_factory()
    async with factory() as db:
        if jwt_payload.get("email") == DEMO_EMAIL:
            user_id = jwt_payload.get("sub")
            since = datetime.now(timezone.utc) - timedelta(days=1)
            result = await db.execute(
                select(func.count(Session.id))
                .where(Session.user_id == user_id, Session.created_at >= since)
            )
            if result.scalar() >= DEMO_DAILY_SESSION_LIMIT:
                await send_message(
                    websocket,
                    WSServerMessage(type=WSMessageType.ERROR, error="Demo account is limited to 10 sessions per day."),
                )
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

        result = await db.execute(
            select(Session)
            .options(
                selectinload(Session.scenario).selectinload(Scenario.persona),
                selectinload(Session.transcript),
            )
            .where(Session.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            await send_message(
                websocket,
                WSServerMessage(type=WSMessageType.ERROR, error="Session not found"),
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        if not session.scenario:
            await send_message(
                websocket,
                WSServerMessage(type=WSMessageType.ERROR, error="Scenario not found"),
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        if session.status == "completed":
            await send_message(
                websocket,
                WSServerMessage(type=WSMessageType.ERROR, error="Session already completed"),
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        persona_ctx = build_persona_context(session.scenario)
        scenario_ctx = build_scenario_context(session.scenario)
        orchestrator = ConversationOrchestrator(persona_ctx, scenario_ctx)
        session.status = "in_progress"
        session.start_time = datetime.now(timezone.utc)
        await db.commit()

        if not session.transcript:
            transcript = Transcript(session_id=session_id, turns=[])
            db.add(transcript)
            await db.commit()

        settings = get_settings()
        live_config = LiveSessionConfig(
            model=settings.live_api_model,
            input_audio_transcription=True,
            output_audio_transcription=True,
        )
        live_client = LiveAPIClient(config=live_config)

        pending_rep_text = ""
        pending_ai_text = ""
        turn_audio_duration_ms = 0

        async def on_live_audio(audio_data: bytes):
            """Forward audio from Gemini Live to client."""
            try:
                await send_message(
                    websocket,
                    WSServerMessage(
                        type=WSMessageType.AUDIO,
                        speaker="ai",
                        data=base64.b64encode(audio_data).decode(),
                    ),
                )
            except Exception as e:
                logger.error(f"Error sending audio to client: {e}")

        async def on_transcription(speaker: str, text: str):
            """Handle transcriptions from Gemini Live."""
            nonlocal pending_rep_text, pending_ai_text

            try:
                display_speaker = "rep" if speaker == "user" else "ai"
                await send_message(
                    websocket,
                    WSServerMessage(
                        type=WSMessageType.TRANSCRIPT,
                        speaker=display_speaker,
                        text=text,
                    ),
                )

                if speaker == "user":
                    pending_rep_text += text + " "
                    
                    if len(pending_rep_text) > 20:
                        quick_result = await orchestrator.gemini.quick_sentiment(pending_rep_text)
                        if quick_result["mood"] != "neutral":
                            mood_map = {
                                "more_interested": "engaged",
                                "less_interested": "bored", 
                                "annoyed": "annoyed",
                                "impressed": "interested",
                                "curious": "curious",
                                "skeptical": "skeptical",
                            }
                            if quick_result["mood"] in mood_map:
                                orchestrator.state.current_mood = mood_map[quick_result["mood"]]
                                orchestrator.state.rapport_level = max(0, min(1,
                                    orchestrator.state.rapport_level + quick_result["rapport_delta"]
                                ))
                                
                                await send_message(
                                    websocket,
                                    WSServerMessage(
                                        type=WSMessageType.STATE_UPDATE,
                                        mood=orchestrator.state.current_mood,
                                        rapport=orchestrator.state.rapport_level,
                                    ),
                                )
                else:
                    pending_ai_text += text + " "

            except Exception as e:
                logger.error(f"Error sending transcription: {e}")

        try:
            system_prompt = orchestrator.get_initial_system_prompt()
            await live_client.connect(
                system_instruction=system_prompt,
                on_audio=on_live_audio,
                on_transcription=on_transcription,
            )
            logger.info(f"Live API connected for session {session_id}")

            await send_message(
                websocket,
                WSServerMessage(
                    type=WSMessageType.CALL_STARTED,
                    session_id=str(session_id),
                    mood=orchestrator.state.current_mood,
                    rapport=orchestrator.state.rapport_level,
                ),
            )

            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == WSMessageType.END_CALL.value:
                    logger.info(f"End call received for session {session_id}")
                    break

                elif msg_type == WSMessageType.AUDIO.value:
                    audio_b64 = data.get("data")
                    if not audio_b64:
                        continue

                    audio_bytes = base64.b64decode(audio_b64)
                    await live_client.send_audio(audio_bytes)
                    turn_audio_duration_ms += len(audio_bytes) // 32

                elif msg_type == "turn_complete":
                    if pending_rep_text.strip() or pending_ai_text.strip():
                        rep_text = pending_rep_text.strip()
                        ai_text = pending_ai_text.strip()
                        duration = turn_audio_duration_ms
                        
                        pending_rep_text = ""
                        pending_ai_text = ""
                        turn_audio_duration_ms = 0
                        
                        async def background_analysis():
                            try:
                                context_injection = await orchestrator.process_turn(
                                    rep_text=rep_text or "(silence)",
                                    ai_text=ai_text or "(no response)",
                                    rep_audio_duration_ms=duration,
                                    ai_audio_duration_ms=len(ai_text) * 50 if ai_text else 0,
                                )
                                if live_client.is_connected:
                                    await live_client.inject_context(context_injection)
                                
                                await send_message(
                                    websocket,
                                    WSServerMessage(
                                        type=WSMessageType.STATE_UPDATE,
                                        mood=orchestrator.state.current_mood,
                                        rapport=orchestrator.state.rapport_level,
                                    ),
                                )
                                logger.debug(f"State update sent: mood={orchestrator.state.current_mood}, rapport={orchestrator.state.rapport_level}")
                            except Exception as e:
                                logger.warning(f"Background analysis failed: {e}")
                        
                        asyncio.create_task(background_analysis())

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")

        except Exception as e:
            logger.error(f"Error in call WebSocket: {e}")
            await send_message(
                websocket,
                WSServerMessage(type=WSMessageType.ERROR, error=str(e)),
            )

        finally:
            await live_client.disconnect()

            if pending_rep_text.strip() or pending_ai_text.strip():
                await orchestrator.process_turn(
                    rep_text=pending_rep_text.strip() or "(end)",
                    ai_text=pending_ai_text.strip() or "(end)",
                    rep_audio_duration_ms=turn_audio_duration_ms,
                )

            await save_call_results(session_id, orchestrator)

            try:
                await send_message(
                    websocket,
                    WSServerMessage(
                        type=WSMessageType.CALL_ENDED,
                        session_id=str(session_id),
                    ),
                )
            except Exception:
                pass


async def save_call_results(session_id: UUID, orchestrator: ConversationOrchestrator):
    """Save all call results: transcript, feedback, analytics, score."""
    factory = get_session_factory()
    async with factory() as db:
        try:
            result = await db.execute(
                select(Session).where(Session.id == session_id)
            )
            session = result.scalar_one_or_none()
            if not session:
                return

            if session.status == "in_progress":
                session.status = "completed"
                session.end_time = datetime.now(timezone.utc)
                if session.start_time:
                    duration = (session.end_time - session.start_time).total_seconds()
                    session.duration_seconds = int(duration)

            feedback_data = await orchestrator.generate_feedback()
            overall_score = Decimal(str(feedback_data.get("overall_score", 50)))
            session.overall_score = overall_score
            
            state = orchestrator.get_state()
            session.rapport_score = Decimal(str(round(state.rapport_level, 2)))

            result = await db.execute(
                select(Transcript).where(Transcript.session_id == session_id)
            )
            transcript = result.scalar_one_or_none()
            if transcript:
                transcript.turns = orchestrator.get_transcript()
            else:
                transcript = Transcript(
                    session_id=session_id,
                    turns=orchestrator.get_transcript(),
                )
                db.add(transcript)

            result = await db.execute(
                select(Feedback).where(Feedback.session_id == session_id)
            )
            existing_feedback = result.scalar_one_or_none()
            if existing_feedback:
                existing_feedback.strengths = feedback_data.get("strengths", [])
                existing_feedback.weaknesses = feedback_data.get("weaknesses", [])
                existing_feedback.suggestions = feedback_data.get("suggestions", [])
                existing_feedback.highlighted_moments = feedback_data.get("highlighted_moments", [])
            else:
                feedback = Feedback(
                    session_id=session_id,
                    strengths=feedback_data.get("strengths", []),
                    weaknesses=feedback_data.get("weaknesses", []),
                    suggestions=feedback_data.get("suggestions", []),
                    highlighted_moments=feedback_data.get("highlighted_moments", []),
                    ai_generated=True,
                )
                db.add(feedback)

            analytics_data = orchestrator.get_analytics()
            result = await db.execute(
                select(SessionAnalytics).where(SessionAnalytics.session_id == session_id)
            )
            existing_analytics = result.scalar_one_or_none()

            analytics_values = {
                "rep_talk_time_seconds": analytics_data.get("rep_talk_time_seconds", 0),
                "ai_talk_time_seconds": analytics_data.get("ai_talk_time_seconds", 0),
                "talk_listen_ratio": Decimal(str(analytics_data.get("talk_listen_ratio", 0.5))),
                "total_turns": analytics_data.get("total_turns", 0),
                "rep_turns": analytics_data.get("rep_turns", 0),
                "ai_turns": analytics_data.get("ai_turns", 0),
                "avg_rep_response_time_ms": analytics_data.get("avg_rep_response_time_ms", 0),
                "behavior_markers": analytics_data.get("behavior_markers", {}),
                "sentiment_timeline": analytics_data.get("sentiment_timeline", []),
                "goal_completion": feedback_data.get("goal_achieved", False),
            }

            if existing_analytics:
                for key, value in analytics_values.items():
                    setattr(existing_analytics, key, value)
            else:
                analytics = SessionAnalytics(
                    session_id=session_id,
                    **analytics_values,
                )
                db.add(analytics)

            state = orchestrator.get_state()
            conversation_state = ConversationState(
                session_id=session_id,
                turn_number=state.turn_count,
                current_mood=state.current_mood,
                rapport_level=Decimal(str(round(state.rapport_level, 2))),
                interest_level=Decimal(str(round(state.interest_level, 2))),
                objections_raised=state.objections_raised,
                rep_strengths_observed=state.rep_strengths_observed,
                rep_weaknesses_observed=state.rep_weaknesses_observed,
                conversation_summary=state.conversation_summary,
                dynamic_modifiers=state.dynamic_modifiers,
            )
            db.add(conversation_state)

            await db.commit()
            logger.info(f"Saved call results for session {session_id}, score: {overall_score}")

        except Exception as e:
            logger.error(f"Error saving call results: {e}")
            await db.rollback()
