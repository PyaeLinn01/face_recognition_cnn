import { useEffect, useState } from 'react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Users } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';

export default function Teachers() {
  const [teachers, setTeachers] = useState<any[]>([]);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTeachers();
  }, []);

  const fetchTeachers = async () => {
    try {
      const [{ data: roleRows, error: roleError }, { data: classData, error: classError }] =
        await Promise.all([
          supabase
            .from('user_roles')
            .select('user_id')
            .eq('role', 'teacher')
            .order('created_at', { ascending: false }),
          supabase
            .from('classes')
            .select('teacher_id'),
        ]);

      if (roleError) throw roleError;
      if (classError) throw classError;

      const counts: Record<string, number> = {};
      (classData || []).forEach((cls) => {
        if (cls.teacher_id) {
          counts[cls.teacher_id] = (counts[cls.teacher_id] || 0) + 1;
        }
      });

      const teacherIds = (roleRows || []).map((r) => r.user_id);

      if (teacherIds.length > 0) {
        const { data: profiles, error: profilesError } = await supabase
          .from('profiles')
          .select('user_id, full_name, email')
          .in('user_id', teacherIds);

        if (profilesError) throw profilesError;

        // keep same order as roles query
        const teachersWithProfile = teacherIds.map((id: string) =>
          profiles?.find((p) => p.user_id === id) || { user_id: id }
        );

        setTeachers(teachersWithProfile);
      } else {
        setTeachers([]);
      }
      setClassCounts(counts);
    } catch (error) {
      console.error('Error fetching teachers:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold font-display">Teachers</h1>
          <p className="text-muted-foreground mt-1">
            View teachers and their assigned classes.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5 text-primary" />
              Teacher Directory
            </CardTitle>
            <CardDescription>Teachers registered in the system.</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">Loading teachers...</div>
            ) : teachers.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No teachers found.
              </div>
            ) : (
              <div className="space-y-3">
                {teachers.map((teacher) => (
                  <div
                    key={teacher.user_id}
                    className="flex items-center justify-between p-4 rounded-xl border border-border"
                  >
                    <div>
                      <p className="font-medium">
                        {teacher.full_name || 'Unknown'}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {teacher.email}
                      </p>
                    </div>
                    <Badge variant="outline">
                      {classCounts[teacher.user_id] || 0} class
                      {classCounts[teacher.user_id] === 1 ? '' : 'es'}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
