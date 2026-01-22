import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Users, CheckCircle, XCircle, Camera, Loader2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

export default function Students() {
  const { toast } = useToast();
  const [students, setStudents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [processingId, setProcessingId] = useState<string | null>(null);

  useEffect(() => {
    fetchStudents();
  }, []);

  const fetchStudents = async () => {
    try {
      const { data: registrations, error } = await supabase
        .from('student_registrations')
        .select(`
          *,
          majors(name),
          classes(name)
        `)
        .order('created_at', { ascending: false });

      if (error) throw error;

      const regs = registrations || [];
      const userIds = regs.map((r) => r.user_id);

      let profilesByUserId: Record<string, any> = {};

      if (userIds.length > 0) {
        const { data: profiles, error: profilesError } = await supabase
          .from('profiles')
          .select('user_id, full_name, email')
          .in('user_id', userIds);

        if (profilesError) throw profilesError;

        profilesByUserId = (profiles || []).reduce(
          (acc, profile) => ({ ...acc, [profile.user_id]: profile }),
          {} as Record<string, any>
        );
      }

      const enriched = regs.map((reg) => ({
        ...reg,
        profiles: profilesByUserId[reg.user_id] || null,
      }));

      setStudents(enriched);
    } catch (error) {
      console.error('Error fetching students:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (id: string) => {
    setProcessingId(id);
    try {
      const { error } = await supabase
        .from('student_registrations')
        .update({ is_approved: true })
        .eq('id', id);

      if (error) throw error;

      toast({
        title: 'Student approved',
        description: 'The student can now mark attendance.',
      });

      fetchStudents();
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: error.message,
      });
    } finally {
      setProcessingId(null);
    }
  };

  const handleReject = async (id: string) => {
    if (!confirm('Are you sure you want to remove this student registration?')) return;
    
    setProcessingId(id);
    try {
      const { error } = await supabase
        .from('student_registrations')
        .delete()
        .eq('id', id);

      if (error) throw error;

      toast({
        title: 'Registration removed',
        description: 'The student registration has been removed.',
      });

      fetchStudents();
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: error.message,
      });
    } finally {
      setProcessingId(null);
    }
  };

  const pendingStudents = students.filter(s => !s.is_approved);
  const approvedStudents = students.filter(s => s.is_approved);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold font-display">Students</h1>
          <p className="text-muted-foreground mt-1">
            Review and manage student registrations.
          </p>
        </div>

        {/* Pending approvals */}
        {pendingStudents.length > 0 && (
          <Card className="border-warning/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-warning">
                <Users className="w-5 h-5" />
                Pending Approvals ({pendingStudents.length})
              </CardTitle>
              <CardDescription>Students waiting for registration approval</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {pendingStudents.map((student, index) => (
                <motion.div
                  key={student.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center justify-between p-4 rounded-xl border border-border bg-card"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-warning/20 flex items-center justify-center">
                      <span className="font-semibold text-warning">
                        {student.profiles?.full_name?.charAt(0) || '?'}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium">{student.profiles?.full_name || 'Unknown'}</p>
                      <p className="text-sm text-muted-foreground">{student.profiles?.email}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          {student.majors?.name}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {student.classes?.name}
                        </Badge>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="success"
                      size="sm"
                      onClick={() => handleApprove(student.id)}
                      disabled={processingId === student.id}
                    >
                      {processingId === student.id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <>
                          <CheckCircle className="w-4 h-4 mr-1" />
                          Approve
                        </>
                      )}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-destructive hover:text-destructive"
                      onClick={() => handleReject(student.id)}
                      disabled={processingId === student.id}
                    >
                      <XCircle className="w-4 h-4 mr-1" />
                      Reject
                    </Button>
                  </div>
                </motion.div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Approved students */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-success" />
              Registered Students ({approvedStudents.length})
            </CardTitle>
            <CardDescription>Students approved and registered in the system</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">Loading students...</div>
            ) : approvedStudents.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No approved students yet.
              </div>
            ) : (
              <div className="space-y-3">
                {approvedStudents.map((student, index) => (
                  <motion.div
                    key={student.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className="flex items-center justify-between p-4 rounded-xl border border-border hover:border-primary/20 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                        <span className="font-semibold text-primary">
                          {student.profiles?.full_name?.charAt(0) || '?'}
                        </span>
                      </div>
                      <div>
                        <p className="font-medium">{student.profiles?.full_name || 'Unknown'}</p>
                        <p className="text-sm text-muted-foreground">{student.profiles?.email}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className="text-xs">
                            {student.majors?.name}
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            {student.classes?.name}
                          </Badge>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {student.face_registered ? (
                        <Badge className="bg-success/10 text-success hover:bg-success/20">
                          <Camera className="w-3 h-3 mr-1" />
                          Face Registered
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-muted-foreground">
                          <Camera className="w-3 h-3 mr-1" />
                          No Face
                        </Badge>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-destructive hover:text-destructive"
                        onClick={() => handleReject(student.id)}
                      >
                        Remove
                      </Button>
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
