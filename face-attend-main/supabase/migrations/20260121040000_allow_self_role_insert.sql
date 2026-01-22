-- Allow users to set their own role during signup (no admin self-assign).
CREATE POLICY "Users can insert their own role"
  ON public.user_roles
  FOR INSERT
  WITH CHECK (
    auth.uid() = user_id
    AND role IN ('student', 'teacher')
    AND NOT EXISTS (
      SELECT 1 FROM public.user_roles ur WHERE ur.user_id = auth.uid()
    )
  );
