import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { GraduationCap, BookOpen, Loader2, CheckCircle } from 'lucide-react';
import { useAuth } from '@/lib/auth-context';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { useNavigate } from 'react-router-dom';

export default function RegisterClass() {
  const { user } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [majors, setMajors] = useState<any[]>([]);
  const [classes, setClasses] = useState<any[]>([]);
  const [selectedMajor, setSelectedMajor] = useState('');
  const [selectedClass, setSelectedClass] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [existingRegistration, setExistingRegistration] = useState<any | null>(null);

  useEffect(() => {
    void fetchMajors();
    void fetchExistingRegistration();
  }, []);

  useEffect(() => {
    if (selectedMajor) {
      void fetchClasses(selectedMajor);
    } else {
      setClasses([]);
      setSelectedClass('');
    }
  }, [selectedMajor]);

  const fetchExistingRegistration = async () => {
    if (!user) return;
    try {
      const { data } = await supabase
        .from('student_registrations')
        .select('id, major_id, class_id, is_approved')
        .eq('user_id', user.id)
        .single();
      if (data) {
        setExistingRegistration(data);
        setSelectedMajor(data.major_id);
        setSelectedClass(data.class_id);
      }
    } catch (error) {
      // no existing registration is fine
    }
  };

  const fetchMajors = async () => {
    try {
      const { data, error } = await supabase
        .from('majors')
        .select('id, name')
        .order('name');

      if (error) throw error;
      setMajors(data || []);
    } catch (error) {
      console.error('Error fetching majors:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchClasses = async (majorId: string) => {
    try {
      const { data, error } = await supabase
        .from('classes')
        .select('id, name')
        .eq('major_id', majorId)
        .order('name');

      if (error) throw error;
      setClasses(data || []);
    } catch (error) {
      console.error('Error fetching classes:', error);
      setClasses([]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user) return;
    if (existingRegistration) {
      toast({
        variant: 'destructive',
        title: 'Registration exists',
        description: 'You already registered for a class. Contact admin to change.',
      });
      return;
    }
    
    setSaving(true);
    try {
      const { error } = await supabase
        .from('student_registrations')
        .insert({
          user_id: user.id,
          major_id: selectedMajor,
          class_id: selectedClass,
          is_approved: false,
          face_registered: false,
        });

      if (error) throw error;

      toast({
        title: 'Registration submitted!',
        description: 'Your registration is pending admin approval.',
      });

      navigate('/dashboard');
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

  return (
    <DashboardLayout>
      <div className="max-w-lg mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold font-display">Register for Class</h1>
          <p className="text-muted-foreground mt-1">
            Select your major and class to get started. You can register only once.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GraduationCap className="w-5 h-5 text-primary" />
              Class Registration
            </CardTitle>
            <CardDescription>
              Your registration will be reviewed by an administrator.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label>Major</Label>
                <Select value={selectedMajor} onValueChange={setSelectedMajor} required disabled={!!existingRegistration}>
                  <SelectTrigger className="h-12">
                    <SelectValue placeholder="Select your major" />
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
                <Label>Class</Label>
                <Select
                  value={selectedClass}
                  onValueChange={setSelectedClass}
                  disabled={!selectedMajor || !!existingRegistration}
                  required
                >
                  <SelectTrigger className="h-12">
                    <SelectValue placeholder={selectedMajor ? "Select your class" : "First select a major"} />
                  </SelectTrigger>
                  <SelectContent>
                    {classes.map((cls) => (
                      <SelectItem key={cls.id} value={cls.id}>
                        {cls.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Button
                type="submit"
                variant="hero"
                className="w-full h-12"
                disabled={saving || !selectedMajor || !selectedClass || !!existingRegistration}
              >
                {saving ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Submit Registration
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
