import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Users, CheckCircle, Clock, BookOpen, Eye } from 'lucide-react';
import { Link } from 'react-router-dom';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/lib/auth-context';

export function TeacherDashboard() {
  const { user } = useAuth();
  const [classes, setClasses] = useState<any[]>([]);
  const [todayAttendance, setTodayAttendance] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchData();
    }
  }, [user]);

  const fetchData = async () => {
    try {
      // Fetch assigned classes
      const { data: classData } = await supabase
        .from('classes')
        .select('*, majors(*)')
        .eq('teacher_id', user?.id);
      
      setClasses(classData || []);

      // Fetch today's attendance
      const today = new Date().toISOString().split('T')[0];
      if (classData && classData.length > 0) {
        const { data: attendance } = await supabase
          .from('attendance_records')
          .select('*, profiles:student_id(full_name)')
          .in('class_id', classData.map(c => c.id))
          .gte('created_at', today);
        
        setTodayAttendance(attendance || []);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const presentToday = todayAttendance.filter(a => a.status === 'present').length;
  const absentToday = todayAttendance.filter(a => a.status === 'absent').length;

  const stats = [
    {
      title: 'My Classes',
      value: classes.length,
      icon: BookOpen,
      color: 'text-primary',
      bgColor: 'bg-primary/10',
    },
    {
      title: 'Present Today',
      value: presentToday,
      icon: CheckCircle,
      color: 'text-success',
      bgColor: 'bg-success/10',
    },
    {
      title: 'Absent Today',
      value: absentToday,
      icon: Clock,
      color: 'text-destructive',
      bgColor: 'bg-destructive/10',
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold font-display">Teacher Dashboard</h1>
        <p className="text-muted-foreground mt-1">Monitor attendance and manage your classes.</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
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
              <Eye className="w-5 h-5 text-primary" />
              Live Attendance
            </CardTitle>
            <CardDescription>Monitor real-time student check-ins</CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/dashboard/live-attendance">
              <Button variant="hero" className="w-full">
                View Live Attendance
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5 text-accent" />
              Manage Attendance
            </CardTitle>
            <CardDescription>Mark absences and update records</CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/dashboard/manage-attendance">
              <Button variant="accent" className="w-full">
                Manage Attendance
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* My Classes */}
      <Card>
        <CardHeader>
          <CardTitle>My Classes</CardTitle>
          <CardDescription>Classes you're assigned to teach</CardDescription>
        </CardHeader>
        <CardContent>
          {classes.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">
              You haven't been assigned to any classes yet.
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {classes.map((cls) => (
                <div
                  key={cls.id}
                  className="p-4 rounded-xl border border-border hover:border-primary/30 transition-colors"
                >
                  <h4 className="font-semibold">{cls.name}</h4>
                  <p className="text-sm text-muted-foreground">{cls.majors?.name}</p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
