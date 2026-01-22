import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Users, GraduationCap, BookOpen, UserCog, Shield, AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';
import { supabase } from '@/integrations/supabase/client';

export function AdminDashboard() {
  const [stats, setStats] = useState({
    majors: 0,
    classes: 0,
    teachers: 0,
    students: 0,
    pendingRegistrations: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const [majors, classes, teachers, students, pending] = await Promise.all([
        supabase.from('majors').select('id', { count: 'exact' }),
        supabase.from('classes').select('id', { count: 'exact' }),
        supabase.from('user_roles').select('id', { count: 'exact' }).eq('role', 'teacher'),
        supabase.from('user_roles').select('id', { count: 'exact' }).eq('role', 'student'),
        supabase.from('student_registrations').select('id', { count: 'exact' }).eq('is_approved', false),
      ]);

      setStats({
        majors: majors.count || 0,
        classes: classes.count || 0,
        teachers: teachers.count || 0,
        students: students.count || 0,
        pendingRegistrations: pending.count || 0,
      });
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Majors',
      value: stats.majors,
      icon: GraduationCap,
      color: 'text-primary',
      bgColor: 'bg-primary/10',
      link: '/dashboard/majors',
    },
    {
      title: 'Total Classes',
      value: stats.classes,
      icon: BookOpen,
      color: 'text-accent',
      bgColor: 'bg-accent/10',
      link: '/dashboard/classes',
    },
    {
      title: 'Teachers',
      value: stats.teachers,
      icon: UserCog,
      color: 'text-success',
      bgColor: 'bg-success/10',
      link: '/dashboard/teachers',
    },
    {
      title: 'Students',
      value: stats.students,
      icon: Users,
      color: 'text-warning',
      bgColor: 'bg-warning/10',
      link: '/dashboard/students',
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold font-display">Admin Dashboard</h1>
        <p className="text-muted-foreground mt-1">Manage your institution's attendance system.</p>
      </div>

      {/* Pending registrations alert */}
      {stats.pendingRegistrations > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="border-warning/50 bg-warning/5">
            <CardContent className="flex items-center gap-4 py-4">
              <div className="w-10 h-10 rounded-full bg-warning/20 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-warning" />
              </div>
              <div>
                <p className="font-medium">{stats.pendingRegistrations} Pending Registrations</p>
                <p className="text-sm text-muted-foreground">
                  Students waiting for approval to join classes.
                </p>
              </div>
              <Link to="/dashboard/students" className="ml-auto">
                <Button variant="accent" size="sm">Review Now</Button>
              </Link>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Link to={stat.link}>
              <Card className="hover:border-primary/30 transition-colors cursor-pointer">
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
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Quick actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GraduationCap className="w-5 h-5 text-primary" />
              Manage Majors
            </CardTitle>
            <CardDescription>Create and edit academic majors</CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/dashboard/majors">
              <Button variant="hero" className="w-full">
                Manage Majors
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-accent" />
              Manage Classes
            </CardTitle>
            <CardDescription>Create classes and assign teachers</CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/dashboard/classes">
              <Button variant="accent" className="w-full">
                Manage Classes
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5 text-success" />
              Student Approvals
            </CardTitle>
            <CardDescription>Review and approve student registrations</CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/dashboard/students">
              <Button variant="success" className="w-full">
                Review Students
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
