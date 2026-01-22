import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Camera, CheckCircle, BarChart3, AlertCircle, BookOpen } from 'lucide-react';
import { Link } from 'react-router-dom';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/lib/auth-context';

export function StudentDashboard() {
  const { user } = useAuth();
  const [registration, setRegistration] = useState<any>(null);
  const [attendanceStats, setAttendanceStats] = useState({ present: 0, total: 0 });
  const [subjectStats, setSubjectStats] = useState({ attended: 0, total: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchData();
    }
  }, [user]);

  const fetchData = async () => {
    try {
      // Fetch registration
      const { data: regData } = await supabase
        .from('student_registrations')
        .select('*, majors(*), classes(*)')
        .eq('user_id', user?.id)
        .single();
      
      setRegistration(regData);

      // Fetch attendance
      const { data: attendance } = await supabase
        .from('attendance_records')
        .select('*')
        .eq('student_id', user?.id);

      if (attendance) {
        const attendedSessions = attendance.filter((a) => a.status === 'present' || a.status === 'late').length;
        setAttendanceStats({ present: attendedSessions, total: attendance.length });
      }

      // Subjects attended / total subjects (based on registered class)
      if (regData?.class_id) {
        const { data: subjects } = await supabase
          .from('subjects')
          .select('id, name, class_id')
          .eq('class_id', regData.class_id);

        const grouped = Object.values(
          (subjects || []).reduce((acc: Record<string, any>, subj: any) => {
            const baseName = subj.name.includes(' (') ? subj.name.split(' (')[0] : subj.name;
            const key = `${baseName}-${subj.class_id}`;
            if (!acc[key]) acc[key] = { id: key, subjectIds: [] as string[] };
            acc[key].subjectIds.push(subj.id);
            return acc;
          }, {})
        );

        const subjectIdsWithAttendance = new Set(
          (attendance || []).map((a: any) => a.subject_id).filter(Boolean)
        );

        const attendedSubjects = grouped.filter((g: any) =>
          g.subjectIds.some((id: string) => subjectIdsWithAttendance.has(id))
        ).length;

        setSubjectStats({ attended: attendedSubjects, total: grouped.length });
      } else {
        setSubjectStats({ attended: 0, total: 0 });
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const attendancePercentage = attendanceStats.total > 0
    ? Math.round((attendanceStats.present / attendanceStats.total) * 100)
    : 0;

  const stats = [
    {
      title: 'Attendance Rate',
      value: `${attendancePercentage}%`,
      icon: BarChart3,
      color: attendancePercentage >= 75 ? 'text-success' : 'text-warning',
      bgColor: attendancePercentage >= 75 ? 'bg-success/10' : 'bg-warning/10',
    },
    {
      title: 'Subjects Attended',
      value: subjectStats.attended,
      icon: CheckCircle,
      color: 'text-primary',
      bgColor: 'bg-primary/10',
    },
    {
      title: 'Total Subjects',
      value: subjectStats.total,
      icon: BookOpen,
      color: 'text-accent',
      bgColor: 'bg-accent/10',
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold font-display">Student Dashboard</h1>
        <p className="text-muted-foreground mt-1">Welcome back! Here's your attendance overview.</p>
      </div>

      {/* Registration status */}
      {!registration?.is_approved && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="border-warning/50 bg-warning/5">
            <CardContent className="flex items-center gap-4 py-4">
              <div className="w-10 h-10 rounded-full bg-warning/20 flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-warning" />
              </div>
              <div>
                <p className="font-medium">Registration Pending</p>
                <p className="text-sm text-muted-foreground">
                  {!registration 
                    ? 'Please register for a class and major to start marking attendance.'
                    : 'Your registration is pending admin approval.'}
                </p>
              </div>
              {!registration && (
                <Link to="/dashboard/register-class" className="ml-auto">
                  <Button variant="accent" size="sm">Register Now</Button>
                </Link>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Face registration prompt */}
      {registration && !registration.face_registered && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="border-primary/50 bg-primary/5">
            <CardContent className="flex items-center gap-4 py-4">
              <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                <Camera className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="font-medium">Face Not Registered</p>
                <p className="text-sm text-muted-foreground">
                  Register your face to enable facial recognition attendance.
                </p>
              </div>
              <Link to="/dashboard/face-register" className="ml-auto">
                <Button variant="hero" size="sm">Register Face</Button>
              </Link>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + index * 0.1 }}
          >
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl ${stat.bgColor} flex items-center justify-center`}>
                    <stat.icon className={`w-6 h-6 ${stat.color}`} />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">{stat.title}</p>
                    <p className={`text-2xl font-bold font-display ${stat.color}`}>{stat.value}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Quick actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="w-5 h-5 text-primary" />
              Mark Attendance
            </CardTitle>
            <CardDescription>Use facial recognition to mark your attendance</CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/dashboard/attendance">
              <Button variant="hero" className="w-full">
                Mark Attendance Now
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-accent" />
              View Details
            </CardTitle>
            <CardDescription>Check your complete attendance history</CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/dashboard/history">
              <Button variant="accent" className="w-full">
                View Attendance History
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
