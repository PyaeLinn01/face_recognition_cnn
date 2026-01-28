-- If there is no admin yet, allow an authenticated user to self-assign admin.
-- This fixes the common setup order: create teacher/student first, then create first admin.

DROP POLICY IF EXISTS "Bootstrap first admin" ON public.user_roles;

CREATE POLICY "Bootstrap first admin"
  ON public.user_roles
  FOR INSERT
  WITH CHECK (
    role = 'admin'
    AND auth.uid() = user_id
    AND NOT EXISTS (
      SELECT 1 FROM public.user_roles ur WHERE ur.role = 'admin'
    )
  );

