# FaceAttend
Smart attendance and class scheduling with Supabase auth, roles, and facial recognition workflow.

## Features
- Supabase authentication with roles (admin, teacher, student) and profile auto-provisioning.
- Class registration (single class per student), admin approvals, grouped subjects with course dates.
- Attendance marking flow (facial recognition placeholder), per-subject attendance with scheduled totals.
- Responsive dashboard (collapsible mobile sidebar), grouped subject views for admin/teacher/student.

## Tech Stack
- React + TypeScript + Vite
- Supabase (auth, Postgres, RLS policies, migrations)
- Tailwind CSS + shadcn/ui + framer-motion

## Prerequisites
- Node.js 18+
- Supabase project with tables from `supabase/migrations` applied.
- Environment variables:
  - `VITE_SUPABASE_URL`
  - `VITE_SUPABASE_PUBLISHABLE_KEY`

## Setup
```sh
npm install
npm run dev          # http://localhost:8080
```

## Build & Lint
```sh
npm run build
npm run lint
```

## Supabase
- Apply migrations in `supabase/migrations` (SQL editor or `supabase db push`).
- Tables: `profiles`, `user_roles`, `majors`, `classes`, `subjects`, `student_registrations`, `attendance_records`.
- Course dates: `subjects.start_date`, `subjects.end_date`.

## Branding
- Tab title and meta updated to FaceAttend.
- Favicon: replace `public/favicon.ico` with your logo; hard refresh to update.

## Notes
- Students: single class registration enforced.
- Attendance stats in “My Attendance” use scheduled session counts (course date range + weekday) vs. attended.
- Admin Subjects groups slots; delete available; edit by recreating.**
