# forest

Canonical forest for coefficient / OR / HR tables from ``Fit.tidy`` and Cox tidy:
``plot_forest(..., layout="table")``. Gallery GLM/Cox figures use that call;
print journals add ``style="bw"``. ``statract.viz.misc.save_prepared_hr_forest`` is
a compatibility adapter (simple frames delegate here; grouped/panel n/event
layouts stay on ``statract.journal_forest``).

::: statract.viz.forest
