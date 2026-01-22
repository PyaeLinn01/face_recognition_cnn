import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart3, CheckCircle, XCircle, Clock, TrendingUp } from 'lucide-react';
import { useAuth } from '@/lib/auth-context';
import { supabase } from '@/integrations/supabase/client';
import { format } from 'date-fns';

export default function MyAttendance() {
  const { user } = useAuth();
  const [records, setRecords] = useState<any[]>([]);
  const [subjects, setSubjects] = useState<any[]>([]);
  const [registration, setRegistration] = useState<any | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const countOccurrencesInRange = (start: string | null, end: string | null, dayOfWeek: number) => {
    if (!start || !end) return 0;
    const startDate = new Date(`${start}T00:00:00`);
    const endDate = new Date(`${end}T00:00:00`);
    if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime()) || startDate > endDate) return 0;

    let count = 0;
    const d = new Date(startDate);
    while (d <= endDate) {
      if (d.getDay() === dayOfWeek) count += 1;
      d.setDate(d.getDate() + 1);
    }
    return count;
  };

  useEffect(() => {
    if (user) {
      fetchAttendance();
    }
  }, [user]);

  const fetchAttendance = async () => {
    try {
      const { data: reg } = await supabase
        .from('student_registrations')
        .select('class_id')
        .eq('user_id', user?.id)
        .single();
      setRegistration(reg || null);

      const [{ data: recordsData }, { data: subjectsData }] = await Promise.all([
        supabase
          .from('attendance_records')
          .select('*, subjects(name, start_date, end_date)')
          .eq('student_id', user?.id)
          .order('created_at', { ascending: false }),
        reg?.class_id
          ? supabase.from('subjects').select('id, name, start_date, end_date, day_of_week, class_id').eq('class_id', reg.class_id)
          : Promise.resolve({ data: [] }),
      ]);

      setRecords(recordsData || []);
      setSubjects(subjectsData || []);
    } catch (error) {
      console.error('Error fetching attendance:', error);
    } finally {
      setLoading(false);
    }
  };

  const stats = {
    present: records.filter(r => r.status === 'present').length,
    absent: records.filter(r => r.status === 'absent').length,
    late: records.filter(r => r.status === 'late').length,
    total: records.length,
  };

  const percentage = stats.total > 0
    ? Math.round((stats.present / stats.total) * 100)
    : 0;

  const groupedSubjectStats = Object.values(
    subjects.reduce((acc: Record<string, any>, subj: any) => {
      const baseName = subj.name.includes(' (') ? subj.name.split(' (')[0] : subj.name;
      const key = `${baseName}-${subj.class_id || ''}`;
      if (!acc[key]) {
        acc[key] = { id: key, baseName, children: [] as any[] };
      }
      acc[key].children.push(subj);
      return acc;
    }, {})
  ).map((group: any) => {
    const childStats = group.children.map((subj: any) => {
      const subjRecs = records.filter((r) => r.subject_id === subj.id);
      const filtered = subjRecs.filter((r) => {
        const d = new Date(r.created_at);
        const afterStart = subj.start_date ? d >= new Date(subj.start_date) : true;
        const beforeEnd = subj.end_date ? d <= new Date(subj.end_date) : true;
        return afterStart && beforeEnd;
      });
      const attended = filtered.filter((r) => r.status === 'present' || r.status === 'late').length;
      const scheduledTotal = countOccurrencesInRange(subj.start_date ?? null, subj.end_date ?? null, subj.day_of_week);
      const pct = scheduledTotal > 0 ? Math.round((attended / scheduledTotal) * 100) : 0;
      return { ...subj, attended, scheduledTotal, pct };
    });
    const attendedSum = childStats.reduce((s: number, c: any) => s + c.attended, 0);
    const scheduledSum = childStats.reduce((s: number, c: any) => s + c.scheduledTotal, 0);
    const pct = scheduledSum > 0 ? Math.round((attendedSum / scheduledSum) * 100) : 0;
    return { ...group, attended: attendedSum, scheduledTotal: scheduledSum, pct, childStats };
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'present':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'absent':
        return <XCircle className="w-4 h-4 text-destructive" />;
      case 'late':
        return <Clock className="w-4 h-4 text-warning" />;
      default:
        return null;
    }
  };

  const getStatusClass = (status: string) => {
    switch (status) {
      case 'present':
        return 'bg-success/10 text-success';
      case 'absent':
        return 'bg-destructive/10 text-destructive';
      case 'late':
        return 'bg-warning/10 text-warning';
      default:
        return '';
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold font-display">My Attendance</h1>
          <p className="text-muted-foreground mt-1">
            View your attendance history and statistics.
          </p>
        </div>

        {/* Stats overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="border-primary/30">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Overall</p>
                    <p className="text-3xl font-bold font-display text-primary">{percentage}%</p>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-primary" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Present</p>
                    <p className="text-3xl font-bold font-display text-success">{stats.present}</p>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-success/10 flex items-center justify-center">
                    <CheckCircle className="w-6 h-6 text-success" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Absent</p>
                    <p className="text-3xl font-bold font-display text-destructive">{stats.absent}</p>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-destructive/10 flex items-center justify-center">
                    <XCircle className="w-6 h-6 text-destructive" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Late</p>
                    <p className="text-3xl font-bold font-display text-warning">{stats.late}</p>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-warning/10 flex items-center justify-center">
                    <Clock className="w-6 h-6 text-warning" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Per-subject attendance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              Subject Attendance
            </CardTitle>
            <CardDescription>
              Attendance by subject within your registered class.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">Loading...</div>
            ) : groupedSubjectStats.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                {registration?.class_id
                  ? 'No subject attendance yet.'
                  : 'Register for a class to see subject attendance.'}
              </div>
            ) : (
              <div className="space-y-3">
                {groupedSubjectStats.map((group: any) => {
                  const isOpen = expanded === group.id;
                  return (
                    <button
                      key={group.id}
                      className="w-full text-left p-4 rounded-xl border border-border hover:border-primary/20 transition-colors"
                      onClick={() => setExpanded(isOpen ? null : group.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">{group.baseName}</p>
                          <p className="text-sm text-muted-foreground">
                            {group.attended}/{group.scheduledTotal} attended
                          </p>
                        </div>
                        <div className="text-lg font-semibold">
                          {group.pct}%
                        </div>
                      </div>
                      {isOpen && (
                        <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                          {group.childStats.map((child: any) => (
                            <div
                              key={child.id}
                              className="flex items-center justify-between p-3 rounded-lg border border-border/50"
                            >
                              <div>
                                <p className="font-medium">{child.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {child.start_date && child.end_date
                                    ? `Course: ${format(new Date(child.start_date), 'PP')} - ${format(new Date(child.end_date), 'PP')}`
                                    : 'Course window: not set'}
                                </p>
                              </div>
                              <div className="text-sm">
                                {child.attended}/{child.scheduledTotal} ({child.pct}%)
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

        {/* Attendance records */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              Attendance History
            </CardTitle>
            <CardDescription>Your recent attendance records</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">Loading...</div>
            ) : records.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No attendance records found.
              </div>
            ) : (
              <div className="space-y-3">
                {records.map((record, index) => (
                  <motion.div
                    key={record.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 rounded-xl border border-border hover:border-primary/20 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      {getStatusIcon(record.status)}
                      <div>
                        <p className="font-medium">{record.subjects?.name || 'General'}</p>
                        <p className="text-sm text-muted-foreground">
                          {format(new Date(record.created_at), 'PPp')}
                        </p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${getStatusClass(record.status)}`}>
                      {record.status}
                    </span>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
