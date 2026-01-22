import { useEffect, useState } from 'react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { BookOpen } from 'lucide-react';
import { useAuth } from '@/lib/auth-context';
import { supabase } from '@/integrations/supabase/client';

const dayLabels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

export default function Subjects() {
  const { user } = useAuth();
  const [subjects, setSubjects] = useState<any[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchSubjects();
    }
  }, [user]);

  const fetchSubjects = async () => {
    try {
      const { data: classData, error: classError } = await supabase
        .from('classes')
        .select('id, name')
        .eq('teacher_id', user?.id);

      if (classError) throw classError;

      if (!classData || classData.length === 0) {
        setSubjects([]);
        return;
      }

      const { data: subjectData, error: subjectError } = await supabase
        .from('subjects')
        .select('id, name, day_of_week, start_time, end_time, classes(name)')
        .in('class_id', classData.map((cls) => cls.id))
        .order('name');

      if (subjectError) throw subjectError;
      setSubjects(subjectData || []);
    } catch (error) {
      console.error('Error fetching subjects:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold font-display">Subjects</h1>
          <p className="text-muted-foreground mt-1">
            Review subjects assigned to your classes.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" />
              Class Subjects
            </CardTitle>
            <CardDescription>Subjects scheduled for your classes.</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">Loading subjects...</div>
            ) : subjects.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No subjects found for your classes yet.
              </div>
            ) : (
              <div className="space-y-3">
                {Object.values(
                  subjects.reduce((acc: Record<string, any>, subj: any) => {
                    const baseName = subj.name.includes(' (') ? subj.name.split(' (')[0] : subj.name;
                    const key = `${baseName}-${subj.class_id}`;
                    if (!acc[key]) {
                      acc[key] = { baseName, className: subj.classes?.name, children: [] };
                    }
                    acc[key].children.push(subj);
                    return acc;
                  }, {})
                ).map((group: any) => {
                  const first = group.children[0];
                  const isOpen = expandedId === `${group.baseName}-${first.class_id}`;
                  return (
                    <button
                      key={`${group.baseName}-${first.class_id}`}
                      type="button"
                      onClick={() =>
                        setExpandedId(isOpen ? null : `${group.baseName}-${first.class_id}`)
                      }
                      className="w-full text-left p-4 rounded-xl border border-border hover:border-primary/20 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">{group.baseName}</p>
                          <p className="text-sm text-muted-foreground">
                            {group.className || 'Unknown class'}
                          </p>
                        </div>
                      </div>
                      {isOpen && (
                        <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                          {group.children.map((child: any) => (
                            <div
                              key={child.id}
                              className="flex items-center justify-between p-3 rounded-lg border border-border/50"
                            >
                              <div className="flex items-center gap-2">
                                <Badge variant="outline">
                                  {dayLabels[child.day_of_week] || `Day ${child.day_of_week}`}
                                </Badge>
                                <span>
                                  {child.start_time} - {child.end_time}
                                </span>
                                <span className="ml-2 text-foreground">{child.name}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
