import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { UserCog, Plus, Trash2, Loader2, Mail } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI, Teacher } from '@/lib/face-api';

export default function Teachers() {
  const { toast } = useToast();
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');

  const fetchTeachers = async () => {
    try {
      const data = await faceAPI.getTeachers();
      setTeachers(data);
    } catch (error) {
      console.error('Failed to fetch teachers:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTeachers();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim() || !newEmail.trim() || !newPassword) return;

    setCreating(true);
    try {
      await faceAPI.createTeacher(newName, newEmail, newPassword);
      toast({ title: 'Success', description: 'Teacher account created successfully' });
      setNewName('');
      setNewEmail('');
      setNewPassword('');
      fetchTeachers();
    } catch (error: any) {
      toast({ variant: 'destructive', title: 'Error', description: error.response?.data?.error || 'Failed to create teacher' });
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (teacherId: string) => {
    if (!confirm('Are you sure you want to delete this teacher?')) return;
    try {
      await faceAPI.deleteTeacher(teacherId);
      toast({ title: 'Success', description: 'Teacher deleted successfully' });
      fetchTeachers();
    } catch (error: any) {
      toast({ variant: 'destructive', title: 'Error', description: error.response?.data?.error || 'Failed to delete teacher' });
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Manage Teachers</h1>
          <p className="text-muted-foreground mt-1">Create and manage teacher accounts.</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Plus className="w-5 h-5 text-primary" />
              Add New Teacher
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="name">Full Name</Label>
                  <Input id="name" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g., John Doe" required />
                </div>
                <div>
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} placeholder="teacher@example.com" required />
                </div>
                <div>
                  <Label htmlFor="password">Password</Label>
                  <Input id="password" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Min 6 characters" required />
                </div>
              </div>
              <Button type="submit" disabled={creating || !newName.trim() || !newEmail.trim() || newPassword.length < 6}>
                {creating ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Plus className="w-4 h-4 mr-2" />}
                Add Teacher
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserCog className="w-5 h-5 text-primary" />
              All Teachers
            </CardTitle>
            <CardDescription>{teachers.length} teachers registered</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex justify-center py-8"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>
            ) : teachers.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No teachers yet. Add one above.</p>
            ) : (
              <div className="space-y-3">
                {teachers.map((teacher, index) => (
                  <motion.div key={teacher.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <UserCog className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-semibold">{teacher.name}</p>
                        <p className="text-sm text-muted-foreground flex items-center gap-1">
                          <Mail className="w-3 h-3" /> {teacher.email}
                        </p>
                      </div>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => handleDelete(teacher.id)}>
                      <Trash2 className="w-4 h-4 text-destructive" />
                    </Button>
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
