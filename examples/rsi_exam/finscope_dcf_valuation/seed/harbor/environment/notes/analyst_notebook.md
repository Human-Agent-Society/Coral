# Analyst notebook — driver forecasting

Running notes while building the coverage model. Unpolished; read with judgement.

## The problem in one line

The DCF engine is only as good as the drivers we feed it. Garbage forward
growth/margin → garbage intrinsic value. So the whole game is forecasting
`rev_growth`, `ebit_margin`, `reinvest` over five years, plus a per-name `wacc`
and a `terminal_growth`.

## What I tried

**v0 — carry-forward (still in `methods/main`).** Take the last observed values,
hold them flat, one house WACC for everyone. Dead simple. It's clearly wrong for
the high-growth names: I'm projecting 25–35% growth for five years on companies
that are obviously decelerating. The valuations on those names come out wildly
high. It's the floor, not the model.

**v1 — per-name trend (`methods/own_history_trend`).** Fit a short slope on each
company's own last few years and extend it. Better on the names with long, clean
histories. But two failure modes I never fixed:
  - short histories (recent IPOs) — three points, mostly noise, the slope is
    meaningless;
  - flagged years in the fit window — an M&A spike drags the whole trend.
On those it's *worse* than carry-forward.

## Leads I started chasing (none settled)

A few threads, none of which I finished or could rank with any confidence:

- **The fast growers clearly don't stay fast.** Holding growth flat, or extending
  a straight line, overshoots them badly — whatever the right model is, it has to
  bend high growth down over the horizon. How much, and whether the speed is the
  same for every name, I never pinned down.
- **Margins don't run away either** — they seem to settle toward some level rather
  than keep climbing, and the level looks different by sector. Didn't quantify it.
- **One name vs. its peers — how much to lean on each?** For the long, clean
  histories a name's own record might carry it; for the thin / recent-IPO names it
  obviously can't and I reached for sector comparables. The deeper open question is
  whether the calibration names' *realized* five-year outcomes reveal something
  that transfers across the universe, or whether each name is really its own
  story. I tested neither properly — could genuinely go either way.
- **The discount rate can't be one flat number.** Riskier names should discount
  harder; `beta_proxy` is the obvious handle, though plenty of names are missing it
  and would need a fallback. I never settled how to turn it into a wacc.
- `terminal_growth` has to stay below `wacc` or the engine refuses it; the realized
  terminals I've seen cluster low, so I've just been clamping.

## Open questions / dead ends

- Reinvestment looked stable across years for most names — I burned a day trying
  to model it cleverly and a robust "last clean value" did about as well. Maybe
  not where the value is.
- A handful of names have a **step change** mid-history that no smooth curve fits.
  Regime shift? Too few to fit individually. Unsolved.
- Don't overfit the calibration names themselves — I once tuned until the visible
  error looked great and it didn't carry over at all. The point is the *pattern*,
  not the names.
