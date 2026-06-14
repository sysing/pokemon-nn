# Experiment Log

| Release | Date | Epochs | Val Acc | Metal | Key Change |
|---------|------|--------|---------|-------|------------|
| [v0.1](v0.1.md) | 2026-06-14 | 7 (best: 6) | 86.5% | 35.0% | Baseline — color augmentations only |
| [v0.2](v0.2.md) | 2026-06-14 | 30 (best: 28) | 81.3% | 60.0% | + RandomGrayscale(p=0.3) — Metal +25pp |
| [v0.3](v0.3.md) | 2026-06-14 | 30 (best: 29) | 94.7% | 80.0% | 4-channel [RGB+Gray] — Metal +45pp |
| [v0.4](v0.4.md) | 2026-06-14 | 30 (best: 29) | 91.9% | 90.5% | Dragon 21→100 cards; sample size illusion busted |

## Template for new experiments

```markdown
## v0.X — Title

**Date**: YYYY-MM-DD
**Dataset**: N images, 10 classes
**Device**: M4 MacBook Air (MPS)

### Changes from previous
- [list changes]

### Results
| Type | Accuracy |
|------|----------|
| ... | ... |

### Insights
- [what we learned]
```
