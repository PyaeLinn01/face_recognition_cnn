import AdminSubjects from './admin/Subjects';
import TeacherSubjects from './teacher/Subjects';
import { useAuth } from '@/lib/auth-context';

export default function SubjectsRouter() {
  const { role } = useAuth();
  if (role === 'admin') return <AdminSubjects />;
  return <TeacherSubjects />;
}
