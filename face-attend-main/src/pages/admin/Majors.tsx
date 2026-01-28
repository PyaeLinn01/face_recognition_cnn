import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { GraduationCap, Plus, Trash2, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI, Major } from '@/lib/face-api';

export default function Majors() {
  const { toast } = useToast();
  const [majors, setMajors] = useState<Major[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDescription, setNewDescription] = useState('');

  const fetchMajors = async () => {
    try {
      const data = await faceAPI.getMajors();
      setMajors(data);
    } catch (error) {
      console.error('Failed to fetch majors:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMajors();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;

    setCreating(true);
    try {
      await faceAPI.createMajor(newName, newDescription);
      toast({ title: 'Success', description: 'Major created successfully' });
      setNewName('');
      setNewDescription('');
      fetchMajors();
    } catch (error: any) {
      toast({ variant: 'destructive', title: 'Error', description: error.response?.data?.error || 'Failed to create major' });
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (majorId: string) => {
    if (!confirm('Are you sure you want to delete this major?')) return;
    try {
      await faceAPI.deleteMajor(majorId);
      toast({ title: 'Success', description: 'Major deleted successfully' });
      fetchMajors();
    } catch (error: any) {
      toast({ variant: 'destructive', title: 'Error', description: error.response?.data?.error || 'Failed to delete major' });
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Manage Majors</h1>
          <p className="text-muted-foreground mt-1">Create and manage academic majors.</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Plus className="w-5 h-5 text-primary" />
              Add New Major
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="flex gap-4">
              <div className="flex-1">
                <Label htmlFor="name">Name</Label>
                <Input id="name" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g., Computer Science" required />
              </div>
              <div className="flex-1">
                <Label htmlFor="description">Description</Label>
                <Input id="description" value={newDescription} onChange={(e) => setNewDescription(e.target.value)} placeholder="Optional description" />
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
              <GraduationCap className="w-5 h-5 text-primary" />
              All Majors
            </CardTitle>
            <CardDescription>{majors.length} majors registered</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex justify-center py-8"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>
            ) : majors.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No majors yet. Add one above.</p>
            ) : (
              <div className="space-y-3">
                {majors.map((major, index) => (
                  <motion.div key={major.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
                    <div>
                      <p className="font-semibold">{major.name}</p>
                      {major.description && <p className="text-sm text-muted-foreground">{major.description}</p>}
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => handleDelete(major.id)}>
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
