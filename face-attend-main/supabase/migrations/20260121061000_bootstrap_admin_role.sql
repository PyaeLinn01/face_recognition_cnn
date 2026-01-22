-- Allow the very first authenticated user to self-assign admin when no roles exist.
CREATE POLICY "Bootstrap first admin"
  ON public.user_roles
  FOR INSERT
  WITH CHECK (
    role = 'admin'
    AND auth.uid() = user_id
    AND NOT EXISTS (SELECT 1 FROM public.user_roles)
  );
