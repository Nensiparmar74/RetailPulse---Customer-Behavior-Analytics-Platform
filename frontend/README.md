# RetailPulse — Fixed Frontend Dashboard

This is a corrected, buildable React/Vite implementation of the RetailPulse analytics interface.

## What was fixed

The original pasted source had several structural and TypeScript/JSX issues:

- The Python `PROJECT_FILES` dictionary was malformed.
- Several filenames and file contents were not quoted as Python strings.
- `MetricCard.tsx` contained broken Tailwind string interpolation such as unquoted class names.
- `App.tsx` and `main.tsx` were missing even though `index.html` referenced `main.tsx`.
- Required Forecast/Cohort/Quality/Insights pages were missing.
- The old artifact layer hardcoded analytics as "verified"; this build removes those fake defaults.
- Analytics now start in a clear "No verified analytics loaded" state.
- JSON artifact imports are validated before being stored in localStorage.
- The UI remains focused on Overview, Segments, Churn, Forecast, Cohorts, Data Quality, and Executive Insights.

## Run locally

```bash
npm install
npm run dev
```

Build:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

## Expected workspace

Place verified analytical output in JSON form and import it through:

`Import Artifacts`

The artifact schema is defined in:

`src/types/analytics.ts`

## Important

This frontend does not invent machine-learning results.

The official Data Science project should generate the real outputs from the actual Online Retail II dataset using Python/Jupyter/scikit-learn/SARIMA or Prophet/SHAP/MLflow.

Then export those verified results to a compatible JSON artifact and load them into this dashboard.

Do not mark fake demo numbers as verified project results.
