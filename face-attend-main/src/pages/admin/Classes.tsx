import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { BookOpen, Plus, Trash2, UserCog, Loader2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

export default function Classes() {
  const { toast } = useToast();
  const [classes, setClasses] = useState<any[]>([]);
  const [majors, setMajors] = useState<any[]>([]);
  const [teachers, setTeachers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState('');
  const [majorId, setMajorId] = useState('');
  const [teacherId, setTeacherId] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [classesRes, majorsRes, teacherRolesRes] = await Promise.all([
        supabase.from('classes').select('*, majors(name)').order('name'),
        supabase.from('majors').select('*').order('name'),
        supabase.from('user_roles').select('user_id').eq('role', 'teacher'),
      ]);

      setClasses(classesRes.data || []);
      setMajors(majorsRes.data || []);

      const teacherIds = (teacherRolesRes.data || []).map((r: any) => r.user_id);

      if (teacherIds.length > 0) {
        const { data: teacherProfiles, error: profilesError } = await supabase
          .from('profiles')
          .select('user_id, full_name, email')
          .in('user_id', teacherIds);

        if (profilesError) throw profilesError;
        setTeachers(teacherProfiles || []);
      } else {
        setTeachers([]);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      const { error } = await supabase
        .from('classes')
        .insert({
          name,
          major_id: majorId,
          teacher_id: teacherId || null,
        });

      if (error) throw error;

      toast({
        title: 'Class created',
        description: `${name} has been added successfully.`,
      });

      setName('');
      setMajorId('');
      setTeacherId('');
      setDialogOpen(false);
      fetchData();
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: error.message,
      });
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string, className: string) => {
    if (!confirm(`Are you sure you want to delete ${className}?`)) return;

    try {
      const { error } = await supabase
        .from('classes')
        .delete()
        .eq('id', id);

      if (error) throw error;

      toast({
        title: 'Class deleted',
        description: `${className} has been removed.`,
      });

      fetchData();
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: error.message,
      });
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold font-display">Classes</h1>
            <p className="text-muted-foreground mt-1">
              Manage classes and assign teachers.
            </p>
          </div>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="hero">
                <Plus className="w-4 h-4 mr-2" />
                Add Class
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Class</DialogTitle>
                <DialogDescription>
                  Add a new class and optionally assign a teacher.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Class Name</Label>
                  <Input
                    id="name"
                    placeholder="e.g., CS-101"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label>Major</Label>
                  <Select value={majorId} onValueChange={setMajorId} required>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a major" />
                    </SelectTrigger>
                    <SelectContent>
                      {majors.map((major) => (
                        <SelectItem key={major.id} value={major.id}>
                          {major.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Teacher (optional)</Label>
                  <Select value={teacherId} onValueChange={setTeacherId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Assign a teacher" />
                    </SelectTrigger>
                    <SelectContent>
                      {teachers.map((teacher) => (
                        <SelectItem key={teacher.user_id} value={teacher.user_id}>
                          {teacher.full_name || teacher.email}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button type="submit" variant="hero" className="w-full" disabled={saving}>
                  {saving ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    'Create Class'
                  )}
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {loading ? (
            <div className="col-span-full text-center py-12 text-muted-foreground">
              Loading classes...
            </div>
          ) : classes.length === 0 ? (
            <div className="col-span-full text-center py-12">
              <BookOpen className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No classes created yet.</p>
            </div>
          ) : (
            classes.map((cls, index) => (
              <motion.div
                key={cls.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card className="hover:border-primary/30 transition-colors">
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div className="w-10 h-10 rounded-xl gradient-accent flex items-center justify-center">
                        <BookOpen className="w-5 h-5 text-accent-foreground" />
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-muted-foreground hover:text-destructive"
                        onClick={() => handleDelete(cls.id, cls.name)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                    <CardTitle className="mt-2">{cls.name}</CardTitle>
                    <CardDescription>{cls.majors?.name}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <UserCog className="w-4 h-4" />
                      <span>
                        {
                          teachers.find((t) => t.user_id === cls.teacher_id)?.full_name ||
                          'No teacher assigned'
                        }
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
