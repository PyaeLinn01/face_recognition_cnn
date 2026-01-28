-- Allow any authenticated user to view profiles of users who have the teacher role.
-- This prevents "Unknown" in the teacher directory when the viewer is not yet recognized as admin.

CREATE POLICY "Authenticated can view teacher profiles"
  ON public.profiles
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1
      FROM public.user_roles ur
      WHERE ur.user_id = profiles.user_id
        AND ur.role = 'teacher'
    )
  );

