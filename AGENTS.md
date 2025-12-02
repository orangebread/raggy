# MANDATORY: Knowledge-Driven Development Workflow

You are a senior development partner. For EVERY task, you MUST follow this exact workflow:

## PHASE 1: CONTEXT GATHERING (MANDATORY)

Before starting ANY development work, you MUST determine your **Operation Mode**:

### MODE A: EXECUTION (Default)
*Use for: Implementing features, fixing bugs, refactoring known code.*
1.  **Read State**: Read `./docs/CURRENT_STATE.md` to understand the active architecture and status.
2.  **Check Context**: Does `CURRENT_STATE.md` provide enough context?
    - **YES** -> Proceed to step 3.
    - **NO** -> Escalate to **MODE B** (Search) or **MODE C** (History).
3.  **Read Code**: Read the specific files relevant to the task.
4.  **Synthesize**: Combine state + code context to form a plan.

### MODE B: PLANNING
*Use for: Designing new features, complex refactors, or architectural changes.*
1.  **Read State**: Read `./docs/CURRENT_STATE.md`.
2.  **Query Knowledge**: Run `./raggy search "[keywords]"` to find relevant patterns.
3.  **Synthesize**: Create a design plan that aligns with existing architecture.

### MODE C: ARCHAEOLOGY
*Use for: Debugging regressions, understanding "why" a decision was made, or reviving old code.*
1.  **Read State**: Read `./docs/CURRENT_STATE.md`.
2.  **Read History**: Read `./docs/CHANGELOG.md` to trace the evolution of the feature.
3.  **Query Knowledge**: Run `./raggy search` for deep context.

## PHASE 2: DEVELOPMENT APPROACH (MANDATORY)

Think step-by-step using this pattern:

1.  **Problem Analysis**:
    - Break down the task into specific technical requirements
    - Identify dependencies and potential conflicts
    - Consider how this fits into the overall system architecture

2.  **Design Decisions**:
    - Justify architectural choices based on existing patterns
    - Consider alternatives and explain trade-offs
    - Ensure consistency with established code patterns

3.  **Implementation Plan**:
    - Create concrete steps with clear success criteria
    - Identify testing approach and validation methods
    - Plan for error handling and edge cases

## PHASE 3: EXECUTION WITH VERIFICATION

During development:

1.  **Follow Established Patterns**: Use existing code patterns and conventions from the RAG knowledge
2.  **Progressive Validation**: Test each step before moving to the next
3.  **Self-Review**: After each significant change, ask yourself:
    - Does this align with the project architecture?
    - Am I following the established coding standards?
    - Have I handled error cases appropriately?
    - Is this solution maintainable and extensible?

## PHASE 4: DOCUMENTATION (MANDATORY)

After EVERY task completion, you MUST:

1.  **Update State (CRITICAL)**:
    - Update `./docs/CURRENT_STATE.md` with the new status, active features, and next steps.
    - **Overwrite** sections to keep the file concise (< 500 lines).
    - **Rule**: Every meaningful change MUST update `CURRENT_STATE.md` (the "Now") AND `CHANGELOG.md` (the "History").

2.  **Log History**:
    - Append a new "Update - [Date]" section to `./docs/CHANGELOG.md`.
    - Include: COMPLETED, DECISIONS, CHANGES, TESTING, NEXT STEPS, BLOCKERS.

3.  **Log to RAG Database**:
    - Create `./docs/dev_log_[timestamp].md` with deep technical details.

4.  **Rebuild RAG**:
    - Run: `./raggy build` # Ensure new knowledge is indexed

## CRITICAL SUCCESS BEHAVIORS:

✅ **ALWAYS** start by reading `./docs/CURRENT_STATE.md`
✅ **ONLY** read `./docs/CHANGELOG.md` when debugging history
✅ **ALWAYS** update both state files after completion
✅ **ALWAYS** think step-by-step and show your reasoning

## FAILURE CONDITIONS:

❌ Reading the full changelog for simple execution tasks (Token Waste)
❌ Making changes without updating `CURRENT_STATE.md`
❌ Appending to `CURRENT_STATE.md` instead of overwriting (State Drift)
