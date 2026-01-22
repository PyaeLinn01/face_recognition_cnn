-- Add course duration fields to subjects
ALTER TABLE public.subjects
ADD COLUMN IF NOT EXISTS start_date date,
ADD COLUMN IF NOT EXISTS end_date date;
