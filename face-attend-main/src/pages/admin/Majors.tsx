import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { GraduationCap, Plus, Trash2, Edit, Loader2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

export default function Majors() {
  const { toast } = useToast();
  const [majors, setMajors] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchMajors();
  }, []);

  const fetchMajors = async () => {
    try {
      const { data, error } = await supabase
        .from('majors')
        .select('*, classes(count)')
        .order('name');
      
      if (error) throw error;
      setMajors(data || []);
    } catch (error) {
      console.error('Error fetching majors:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      const { error } = await supabase
        .from('majors')
        .insert({ name, description });

      if (error) throw error;

      toast({
        title: 'Major created',
        description: `${name} has been added successfully.`,
      });

      setName('');
      setDescription('');
      setDialogOpen(false);
      fetchMajors();
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

  const handleDelete = async (id: string, majorName: string) => {
    if (!confirm(`Are you sure you want to delete ${majorName}?`)) return;

    try {
      const { error } = await supabase
        .from('majors')
        .delete()
        .eq('id', id);

      if (error) throw error;

      toast({
        title: 'Major deleted',
        description: `${majorName} has been removed.`,
      });

      fetchMajors();
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
            <h1 className="text-3xl font-bold font-display">Majors</h1>
            <p className="text-muted-foreground mt-1">
              Manage academic majors and departments.
            </p>
          </div>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="hero">
                <Plus className="w-4 h-4 mr-2" />
                Add Major
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Major</DialogTitle>
                <DialogDescription>
                  Add a new academic major to the system.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Major Name</Label>
                  <Input
                    id="name"
                    placeholder="e.g., Computer Science"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">Description (optional)</Label>
                  <Textarea
                    id="description"
                    placeholder="Brief description of the major..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={3}
                  />
                </div>
                <Button type="submit" variant="hero" className="w-full" disabled={saving}>
                  {saving ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    'Create Major'
                  )}
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {loading ? (
            <div className="col-span-full text-center py-12 text-muted-foreground">
              Loading majors...
            </div>
          ) : majors.length === 0 ? (
            <div className="col-span-full text-center py-12">
              <GraduationCap className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No majors created yet.</p>
            </div>
          ) : (
            majors.map((major, index) => (
              <motion.div
                key={major.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card className="hover:border-primary/30 transition-colors">
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                        <GraduationCap className="w-5 h-5 text-primary-foreground" />
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-muted-foreground hover:text-destructive"
                        onClick={() => handleDelete(major.id, major.name)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                    <CardTitle className="mt-2">{major.name}</CardTitle>
                    {major.description && (
                      <CardDescription>{major.description}</CardDescription>
                    )}
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      {major.classes?.[0]?.count || 0} classes
                    </p>
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
