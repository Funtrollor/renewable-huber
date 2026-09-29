# Design notes

These are the engineering records behind the native engines and the
maintainability work. They are kept as written, in their original language,
because other documents, tests and the changelog cite them; read them as the
reasoning at the time a decision was made, and the [API](../api.md) and
[support matrix](../support-matrix.md) as the current contract.

!!! warning "Two numbering schemes"

    `P0`–`P3` means two different things in this repository. Always check
    which scheme a document is using.

    | | Maintainability refactor | Native-core notes |
    | --- | --- | --- |
    | P0 | C ABI contract, cross-language manifest, `engine.cu` split | pure-Python baseline |
    | P1 | Python backend capability contract | Rust CPU engine |
    | P2 | Rust module split | CUDA whole-batch engine |

## Native core

- [Native-core RFC](../native-core-rfc.md): the Rust and CUDA core, its
  compatibility contract, engine rules, benchmark protocol and delivery plan.
- [P0: baseline](../native-core-p0-baseline.md): the historical pre-native
  correctness and performance capture the native engines were compared
  against; not a description of any release.
- [P1: Rust CPU engine](../native-core-p1.md)
- [P2: CUDA whole-batch engine](../native-core-p2.md)
- [P4: CUDA tuning](../native-core-p4.md)
- [Native penalty completion plan](../native-penalty-completion-plan.md):
  `penalty="l1"` on the native CUDA engine (C ABI 2 / Python API 4).

## Dispatch and maintainability

- [CPU auto-dispatch RFC](../cpu-auto-dispatch-rfc.md): when `backend="auto"`
  may select the Rust CPU engine, from bounded runtime evidence.
- [Maintainability refactor](../maintainability-refactor.md) (中文): what the
  P0–P3 audit phases changed and why.

Repository-level working rules, including the invariants that break silently,
are in
[`AGENTS.md`](https://github.com/Funtrollor/renewable-huber/blob/main/AGENTS.md).
