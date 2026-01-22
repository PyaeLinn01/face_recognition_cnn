import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Users, CheckCircle, Clock, Activity } from 'lucide-react';
import { useAuth } from '@/lib/auth-context';
import { supabase } from '@/integrations/supabase/client';
import { format } from 'date-fns';

export default function LiveAttendance() {
  const { user } = useAuth();
  const [classes, setClasses] = useState<any[]>([]);
  const [selectedClass, setSelectedClass] = useState('');
  const [attendance, setAttendance] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchClasses();
    }
  }, [user]);

  useEffect(() => {
    if (selectedClass) {
      fetchAttendance();
      
      // Subscribe to realtime updates
      const channel = supabase
        .channel('attendance-updates')
        .on(
          'postgres_changes',
          {
            event: 'INSERT',
            schema: 'public',
            table: 'attendance_records',
            filter: `class_id=eq.${selectedClass}`,
          },
          (payload) => {
            fetchAttendance();
          }
        )
        .subscribe();

      return () => {
        supabase.removeChannel(channel);
      };
    }
  }, [selectedClass]);

  const fetchClasses = async () => {
    try {
      const { data } = await supabase
        .from('classes')
        .select('*, majors(name)')
        .eq('teacher_id', user?.id);
      
      setClasses(data || []);
      if (data && data.length > 0) {
        setSelectedClass(data[0].id);
      }
    } catch (error) {
      console.error('Error fetching classes:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchAttendance = async () => {
    if (!selectedClass) return;
    
    const today = new Date().toISOString().split('T')[0];
    
    try {
      const { data } = await supabase
        .from('attendance_records')
        .select('*, profiles:student_id(full_name, email)')
        .eq('class_id', selectedClass)
        .gte('created_at', today)
        .order('created_at', { ascending: false });
      
      setAttendance(data || []);
    } catch (error) {
      console.error('Error fetching attendance:', error);
    }
  };

  const presentCount = attendance.filter(a => a.status === 'present').length;

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold font-display">Live Attendance</h1>
            <p className="text-muted-foreground mt-1">
              Monitor real-time student check-ins.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-success animate-pulse" />
            <span className="text-sm text-muted-foreground">Live</span>
          </div>
        </div>

        {classes.length > 0 && (
          <div className="w-full max-w-xs">
            <Select value={selectedClass} onValueChange={setSelectedClass}>
              <SelectTrigger>
                <SelectValue placeholder="Select a class" />
              </SelectTrigger>
              <SelectContent>
                {classes.map((cls) => (
                  <SelectItem key={cls.id} value={cls.id}>
                    {cls.name} - {cls.majors?.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="border-success/30">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-success/10 flex items-center justify-center">
                  <CheckCircle className="w-6 h-6 text-success" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Present Today</p>
                  <p className="text-3xl font-bold font-display text-success">{presentCount}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                  <Users className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Total Check-ins</p>
                  <p className="text-3xl font-bold font-display text-primary">{attendance.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Live feed */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              Live Check-ins
            </CardTitle>
            <CardDescription>Students who have marked attendance today</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">Loading...</div>
            ) : attendance.length === 0 ? (
              <div className="text-center py-12">
                <Clock className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No check-ins yet today.</p>
                <p className="text-sm text-muted-foreground">Students will appear here as they mark attendance.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {attendance.map((record, index) => (
                  <motion.div
                    key={record.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 rounded-xl border border-border bg-card"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-success/20 flex items-center justify-center">
                        <CheckCircle className="w-5 h-5 text-success" />
                      </div>
                      <div>
                        <p className="font-medium">{record.profiles?.full_name || 'Unknown'}</p>
                        <p className="text-sm text-muted-foreground">{record.profiles?.email}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge className="bg-success/10 text-success hover:bg-success/20">Present</Badge>
                      <p className="text-xs text-muted-foreground mt-1">
                        {format(new Date(record.created_at), 'HH:mm')}
                      </p>
                    </div>
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
