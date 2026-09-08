# TODO / leftover leads

Notes-to-self from the prior analyst. Not all of these panned out — treat them as
leads, not gospel.

- [ ] The carry-forward model in `methods/main` is a placeholder. Holding the last
      year flat clearly overvalues the fast growers — nobody grows 30% for five
      straight years. Some kind of decay is the obvious first fix.
- [ ] `methods/own_history_trend` fits each name's own recent slope. It helps on the
      long, clean histories but goes haywire on the short ones and on names with a
      flagged year in the window. I never reconciled that.
- [ ] The calibration names come with their realized outcomes — I keep meaning to
      use them to actually *test* whether the way growth slows and margins settle is
      shared across the universe or really name-by-name. Never ran it cleanly; could
      go either way, and it changes the whole approach.
- [ ] Thin-history names (recent IPOs) have almost nothing to fit. Leaning on
      sector peers seemed to help but I didn't wire it in properly.
- [ ] The discount rate is currently one flat number for everyone. That can't be
      right — riskier names should discount harder, and `beta_proxy` (in
      coverage.csv for most names) is the obvious lever; I never settled how to use it.
- [ ] Drop the special-items years before fitting any trend — they wrecked my
      margin lines until I started excluding them. (See data_dictionary.)
- [ ] HUNCH (unverified): reinvestment is where the value leakage is — maybe spend
      the effort modelling reinvest carefully. (Honestly not sure this moved much.)
- [ ] The off-calendar fiscal names (fy_end_month != 12) might need their history
      realigned to a common calendar before anything else. Never got to it; not
      sure it actually matters for the drivers we forecast.
- [ ] A few names just behave differently partway through the history — a step in
      growth or margin that doesn't fit a smooth curve. Couldn't tell if that's
      noise or something real.
