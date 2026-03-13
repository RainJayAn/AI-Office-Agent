from dataclasses import dataclass

from app.core.config import get_settings


@dataclass(frozen=True)
class ModelProfile:
    name: str
    cost_efficiency: float
    latency_efficiency: float
    reasoning_capability: float


@dataclass(frozen=True)
class TaskRequirements:
    prompt_length: int = 0
    message_count: int = 1
    tool_iteration_count: int = 0
    use_rag: bool = False
    stream: bool = False
    requires_reasoning: bool = False


def choose_chat_model_name(requirements: TaskRequirements | None = None) -> str:
    settings = get_settings()
    task = requirements or TaskRequirements()
    profiles = _get_model_profiles()
    weights = _build_priority_weights(task)

    best_profile = max(
        profiles,
        key=lambda profile: (
            weights["cost"] * profile.cost_efficiency
            + weights["latency"] * profile.latency_efficiency
            + weights["reasoning"] * profile.reasoning_capability
        ),
    )
    return best_profile.name


def _get_model_profiles() -> list[ModelProfile]:
    settings = get_settings()
    return [
        ModelProfile(
            name=settings.MODEL_TURBO_NAME,
            cost_efficiency=1.0,
            latency_efficiency=1.0,
            reasoning_capability=0.45,
        ),
        ModelProfile(
            name=settings.MODEL_PLUS_NAME,
            cost_efficiency=0.65,
            latency_efficiency=0.70,
            reasoning_capability=0.85,
        ),
        ModelProfile(
            name=settings.MODEL_MAX_NAME,
            cost_efficiency=0.15,
            latency_efficiency=0.20,
            reasoning_capability=1.0,
        ),
    ]


def _build_priority_weights(task: TaskRequirements) -> dict[str, float]:
    cost_weight = 0.34
    latency_weight = 0.33
    reasoning_weight = 0.33

    if task.stream:
        latency_weight += 0.35
        cost_weight += 0.10
        reasoning_weight -= 0.45

    if (
        task.prompt_length < 300
        and task.message_count <= 2
        and task.tool_iteration_count == 0
        and not task.use_rag
        and not task.requires_reasoning
    ):
        cost_weight += 0.20
        latency_weight += 0.15
        reasoning_weight -= 0.35

    if task.use_rag:
        reasoning_weight += 0.20
        cost_weight -= 0.10
        latency_weight -= 0.10

    if task.tool_iteration_count >= 1:
        reasoning_weight += 0.15
        latency_weight -= 0.05
        cost_weight -= 0.10

    if task.tool_iteration_count >= 2:
        reasoning_weight += 0.20
        latency_weight -= 0.05
        cost_weight -= 0.10

    if task.requires_reasoning:
        reasoning_weight += 0.35
        cost_weight -= 0.15
        latency_weight -= 0.20

    if task.prompt_length > 2000:
        reasoning_weight += 0.20
        cost_weight -= 0.10
        latency_weight -= 0.10

    if task.message_count > 8:
        reasoning_weight += 0.15
        cost_weight -= 0.10
        latency_weight -= 0.05

    cost_weight = max(cost_weight, 0.05)
    latency_weight = max(latency_weight, 0.05)
    reasoning_weight = max(reasoning_weight, 0.05)

    total = cost_weight + latency_weight + reasoning_weight
    return {
        "cost": cost_weight / total,
        "latency": latency_weight / total,
        "reasoning": reasoning_weight / total,
    }
