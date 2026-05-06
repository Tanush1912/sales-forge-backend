"""Session management routes."""

from datetime import datetime, timezone, timedelta
from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db
from app.models.database import Session, Scenario, Transcript, Feedback, User, ConversationState

DEMO_EMAIL = "demo@salesforge.app"
DEMO_DAILY_SESSION_LIMIT = 10
from app.models.schemas import (
    SessionCreate,
    SessionResponse,
    SessionDetail,
    SessionStatus,
    ScenarioCard,
    PersonaSummary,
    TranscriptResponse,
    FeedbackResponse,
    ErrorResponse,
)
from app.api.routes.auth import get_current_user

router = APIRouter()


@router.get(
    "",
    response_model=List[SessionResponse],
)
async def list_sessions(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: SessionStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[SessionResponse]:
    """List user's sessions with optional filtering."""
    query = (
        select(Session)
        .options(
            selectinload(Session.scenario).selectinload(Scenario.persona),
            selectinload(Session.conversation_states),
        )
        .where(Session.user_id == current_user.id)
        .order_by(Session.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    if status_filter:
        query = query.where(Session.status == status_filter.value)

    result = await db.execute(query)
    sessions = result.scalars().all()

    responses = []
    for session in sessions:
        scenario_card = None
        if session.scenario:
            persona_summary = None
            if session.scenario.persona:
                persona_summary = PersonaSummary(
                    id=session.scenario.persona.id,
                    name=session.scenario.persona.name,
                    title=session.scenario.persona.title,
                    company=session.scenario.persona.company,
                    traits=session.scenario.persona.traits,
                    default_mood=session.scenario.persona.default_mood,
                )
            scenario_card = ScenarioCard(
                id=session.scenario.id,
                name=session.scenario.name,
                type=session.scenario.type,
                difficulty=session.scenario.difficulty,
                category=session.scenario.category,
                book_rate=session.scenario.book_rate,
                is_locked=session.scenario.is_locked,
                persona=persona_summary,
            )

        rapport_score = session.rapport_score
        if rapport_score is None and session.conversation_states:
            latest_state = max(session.conversation_states, key=lambda s: s.turn_number)
            rapport_score = latest_state.rapport_level

        responses.append(
            SessionResponse(
                id=session.id,
                user_id=session.user_id,
                scenario_id=session.scenario_id,
                training_assignment_id=session.training_assignment_id,
                status=session.status,
                start_time=session.start_time,
                end_time=session.end_time,
                duration_seconds=session.duration_seconds,
                overall_score=session.overall_score,
                rapport_score=rapport_score,
                scenario=scenario_card,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        )

    return responses


@router.get(
    "/{session_id}",
    response_model=SessionDetail,
    responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
async def get_session(
    session_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionDetail:
    """Get session details with transcript and feedback."""
    result = await db.execute(
        select(Session)
        .options(
            selectinload(Session.scenario).selectinload(Scenario.persona),
            selectinload(Session.transcript),
            selectinload(Session.feedback),
            selectinload(Session.conversation_states),
        )
        .where(Session.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.user_id != current_user.id and current_user.role == "rep":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view this session",
        )

    scenario_card = None
    if session.scenario:
        persona_summary = None
        if session.scenario.persona:
            persona_summary = PersonaSummary(
                id=session.scenario.persona.id,
                name=session.scenario.persona.name,
                title=session.scenario.persona.title,
                company=session.scenario.persona.company,
                traits=session.scenario.persona.traits,
                default_mood=session.scenario.persona.default_mood,
            )
        scenario_card = ScenarioCard(
            id=session.scenario.id,
            name=session.scenario.name,
            type=session.scenario.type,
            difficulty=session.scenario.difficulty,
            category=session.scenario.category,
            book_rate=session.scenario.book_rate,
            is_locked=session.scenario.is_locked,
            persona=persona_summary,
        )

    transcript_response = None
    if session.transcript:
        transcript_response = TranscriptResponse(
            id=session.transcript.id,
            session_id=session.transcript.session_id,
            turns=session.transcript.turns,
            raw_audio_url=session.transcript.raw_audio_url,
            created_at=session.transcript.created_at,
        )

    feedback_response = None
    if session.feedback:
        feedback_response = FeedbackResponse(
            id=session.feedback.id,
            session_id=session.feedback.session_id,
            strengths=session.feedback.strengths,
            weaknesses=session.feedback.weaknesses,
            suggestions=session.feedback.suggestions,
            highlighted_moments=session.feedback.highlighted_moments,
            ai_generated=session.feedback.ai_generated,
            manager_comments=session.feedback.manager_comments,
            created_at=session.feedback.created_at,
        )

    rapport_score = session.rapport_score
    if rapport_score is None and session.conversation_states:
        latest_state = max(session.conversation_states, key=lambda s: s.turn_number)
        rapport_score = latest_state.rapport_level

    return SessionDetail(
        id=session.id,
        user_id=session.user_id,
        scenario_id=session.scenario_id,
        training_assignment_id=session.training_assignment_id,
        status=session.status,
        start_time=session.start_time,
        end_time=session.end_time,
        duration_seconds=session.duration_seconds,
        overall_score=session.overall_score,
        rapport_score=rapport_score,
        created_at=session.created_at,
        updated_at=session.updated_at,
        scenario=scenario_card,
        transcript=transcript_response,
        feedback=feedback_response,
    )


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
)
async def create_session(
    session_data: SessionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionResponse:
    """Start a new practice session."""
    if current_user.email == DEMO_EMAIL:
        since = datetime.now(timezone.utc) - timedelta(days=1)
        result = await db.execute(
            select(func.count(Session.id))
            .where(Session.user_id == current_user.id, Session.created_at >= since)
        )
        if result.scalar() >= DEMO_DAILY_SESSION_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Demo account is limited to 10 sessions per day. Sign up for full access.",
            )

    result = await db.execute(
        select(Scenario).where(
            Scenario.id == session_data.scenario_id,
            Scenario.is_active == True,
        )
    )
    scenario = result.scalar_one_or_none()

    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario not found or not active",
        )

    if scenario.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This scenario is locked",
        )

    session = Session(
        user_id=current_user.id,
        scenario_id=session_data.scenario_id,
        training_assignment_id=session_data.training_assignment_id,
        status="pending",
    )
    
    transcript = Transcript(
        session=session,
        turns=[],
    )
    db.add(session)
    db.add(transcript)

    await db.flush()
    
    result = await db.execute(
        select(Session)
        .options(selectinload(Session.scenario).selectinload(Scenario.persona))
        .where(Session.id == session.id)
    )
    session = result.scalar_one()

    scenario_card = None
    if session.scenario:
        persona_summary = None
        if session.scenario.persona:
            persona_summary = PersonaSummary(
                id=session.scenario.persona.id,
                name=session.scenario.persona.name,
                title=session.scenario.persona.title,
                company=session.scenario.persona.company,
                traits=session.scenario.persona.traits,
                default_mood=session.scenario.persona.default_mood,
            )
        scenario_card = ScenarioCard(
            id=session.scenario.id,
            name=session.scenario.name,
            type=session.scenario.type,
            difficulty=session.scenario.difficulty,
            category=session.scenario.category,
            book_rate=session.scenario.book_rate,
            is_locked=session.scenario.is_locked,
            persona=persona_summary,
        )

    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        scenario_id=session.scenario_id,
        training_assignment_id=session.training_assignment_id,
        status=session.status,
        start_time=session.start_time,
        end_time=session.end_time,
        duration_seconds=session.duration_seconds,
        overall_score=session.overall_score,
        rapport_score=None,
        scenario=scenario_card,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.patch(
    "/{session_id}/start",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def start_session(
    session_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionResponse:
    """Mark session as started."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start session with status: {session.status}",
        )

    session.status = "in_progress"
    session.start_time = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(session)

    return SessionResponse.model_validate(session)


@router.patch(
    "/{session_id}/end",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def end_session(
    session_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    abandoned: bool = Query(default=False, description="Mark session as abandoned"),
) -> SessionResponse:
    """End an in-progress session."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.status not in ("pending", "in_progress"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot end session with status: {session.status}",
        )

    session.status = "abandoned" if abandoned else "completed"
    session.end_time = datetime.now(timezone.utc)

    if session.start_time:
        duration = (session.end_time - session.start_time).total_seconds()
        session.duration_seconds = int(duration)

    await db.flush()
    await db.refresh(session)

    return SessionResponse.model_validate(session)


@router.get(
    "/stats/summary",
    response_model=dict,
)
async def get_session_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get summary statistics for current user's sessions."""
    result = await db.execute(
        select(
            Session.status,
            func.count(Session.id).label("count"),
        )
        .where(Session.user_id == current_user.id)
        .group_by(Session.status)
    )
    status_counts = {row.status: row.count for row in result}

    result = await db.execute(
        select(func.avg(Session.overall_score))
        .where(
            Session.user_id == current_user.id,
            Session.overall_score.isnot(None),
        )
    )
    avg_score = result.scalar()

    return {
        "total_sessions": sum(status_counts.values()),
        "completed_sessions": status_counts.get("completed", 0),
        "in_progress_sessions": status_counts.get("in_progress", 0),
        "abandoned_sessions": status_counts.get("abandoned", 0),
        "average_score": float(avg_score) if avg_score else None,
    }