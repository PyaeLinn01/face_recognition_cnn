import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { BookOpen, Plus, Trash2, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI, Subject } from '@/lib/face-api';

export default function Subjects() {
  const { toast } = useToast();
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [newCode, setNewCode] = useState('');

  const fetchSubjects = async () => {
    try {
      const data = await faceAPI.getSubjects();
      setSubjects(data);
    } catch (error) {
      console.error('Failed to fetch subjects:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSubjects();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;

    setCreating(true);
    try {
      await faceAPI.createSubject(newName, newCode);
      toast({ title: 'Success', description: 'Subject created successfully' });
      setNewName('');
      setNewCode('');
      fetchSubjects();
    } catch (error: any) {
      toast({ variant: 'destructive', title: 'Error', description: error.response?.data?.error || 'Failed to create subject' });
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (subjectId: string) => {
    if (!confirm('Are you sure you want to delete this subject?')) return;
    try {
      await faceAPI.deleteSubject(subjectId);
      toast({ title: 'Success', description: 'Subject deleted successfully' });
      fetchSubjects();
    } catch (error: any) {
      toast({ variant: 'destructive', title: 'Error', description: error.response?.data?.error || 'Failed to delete subject' });
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Manage Subjects</h1>
          <p className="text-muted-foreground mt-1">Create and manage course subjects.</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Plus className="w-5 h-5 text-primary" />
              Add New Subject
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="flex gap-4">
              <div className="flex-1">
                <Label htmlFor="name">Subject Name</Label>
                <Input id="name" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g., Data Structures" required />
              </div>
              <div className="flex-1">
                <Label htmlFor="code">Subject Code</Label>
                <Input id="code" value={newCode} onChange={(e) => setNewCode(e.target.value)} placeholder="e.g., CS201" />
              </div>
              <div className="flex items-end">
                <Button type="submit" disabled={creating || !newName.trim()}>
                  {creating ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Plus className="w-4 h-4 mr-2" />}
                  Add
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" />
              All Subjects
            </CardTitle>
            <CardDescription>{subjects.length} subjects registered</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex justify-center py-8"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>
            ) : subjects.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No subjects yet. Add one above.</p>
            ) : (
              <div className="space-y-3">
                {subjects.map((subject, index) => (
                  <motion.div key={subject.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
                    <div>
                      <p className="font-semibold">{subject.name}</p>
                      {subject.code && <p className="text-sm text-muted-foreground">Code: {subject.code}</p>}
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => handleDelete(subject.id)}>
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
