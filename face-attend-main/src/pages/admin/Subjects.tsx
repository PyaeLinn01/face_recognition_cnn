import { useEffect, useMemo, useState } from 'react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { BookOpen, Plus, Loader2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

type Option = { value: string; label: string };

const dayOptions: Option[] = [
  { value: '1', label: 'Monday' },
  { value: '2', label: 'Tuesday' },
  { value: '3', label: 'Wednesday' },
  { value: '4', label: 'Thursday' },
  { value: '5', label: 'Friday' },
];

const periodOptions = [
  { id: 'p1', label: '8:30 - 9:30', start_time: '08:30', end_time: '09:30' },
  { id: 'p2', label: '9:40 - 10:40', start_time: '09:40', end_time: '10:40' },
  { id: 'p3', label: '10:50 - 11:50', start_time: '10:50', end_time: '11:50' },
  { id: 'p4', label: '12:40 - 1:40', start_time: '12:40', end_time: '13:40' },
  { id: 'p5', label: '1:50 - 2:50', start_time: '13:50', end_time: '14:50' },
  { id: 'p6', label: '3:00 - 4:00', start_time: '15:00', end_time: '16:00' },
];

export default function Subjects() {
  const { toast } = useToast();
  const [subjects, setSubjects] = useState<any[]>([]);
  const [classes, setClasses] = useState<any[]>([]);
  const [teachers, setTeachers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  const [name, setName] = useState('');
  const [classId, setClassId] = useState('');
  const [teacherId, setTeacherId] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [slots, setSlots] = useState<
    { label: 'First Lecture' | 'Second Lecture' | 'First TDA' | 'Second TDA'; day: string; periodId: string }[]
  >([
    { label: 'First Lecture', day: '', periodId: '' },
    { label: 'Second Lecture', day: '', periodId: '' },
    { label: 'First TDA', day: '', periodId: '' },
    { label: 'Second TDA', day: '', periodId: '' },
  ]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    void fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [{ data: classRows }, { data: teacherRoleRows }, { data: subjectRows }] = await Promise.all([
        supabase.from('classes').select('id, name, majors(name), teacher_id').order('name'),
        supabase.from('user_roles').select('user_id').eq('role', 'teacher'),
        supabase
          .from('subjects')
          .select('*, classes(name, teacher_id)')
          .order('day_of_week')
          .order('start_time'),
      ]);

      setClasses(classRows || []);

      const teacherIds = (teacherRoleRows || []).map((r) => r.user_id);
      if (teacherIds.length > 0) {
        const { data: teacherProfiles } = await supabase
          .from('profiles')
          .select('user_id, full_name, email')
          .in('user_id', teacherIds);
        setTeachers(teacherProfiles || []);
      } else {
        setTeachers([]);
      }

      setSubjects(subjectRows || []);
    } catch (error) {
      console.error('Error fetching subjects:', error);
    } finally {
      setLoading(false);
    }
  };

  const classOptions = useMemo(
    () => classes.map((cls) => ({ value: cls.id, label: `${cls.name} ${cls.majors ? `(${cls.majors.name})` : ''}` })),
    [classes]
  );

  const teacherOptions = useMemo(
    () => teachers.map((t) => ({ value: t.user_id, label: t.full_name || t.email || 'Unknown' })),
    [teachers]
  );

  const updateSlot = (index: number, key: 'day' | 'periodId', value: string) => {
    setSlots((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], [key]: value };
      return next;
    });
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !classId || !teacherId || !startDate || !endDate || slots.some((s) => !s.day || !s.periodId)) {
      toast({
        variant: 'destructive',
        title: 'Missing info',
        description: 'Fill all fields and choose day/period for each slot (2 Lectures, 2 TDA).',
      });
      return;
    }

    setSaving(true);
    try {
      // ensure no conflicts for same class/day/time
      const conflicts: string[] = [];
      for (const slot of slots) {
        const period = periodOptions.find((p) => p.id === slot.periodId)!;
        const { data: existing, error: conflictError } = await supabase
          .from('subjects')
          .select('id')
          .eq('class_id', classId)
          .eq('day_of_week', Number(slot.day))
          .eq('start_time', period.start_time);
        if (conflictError) throw conflictError;
        if (existing && existing.length > 0) {
          conflicts.push(`${slot.label} (${dayOptions.find((d) => d.value === slot.day)?.label} ${period.label})`);
        }
      }

      if (conflicts.length > 0) {
        toast({
          variant: 'destructive',
          title: 'Schedule conflict',
          description: `${conflicts.join(', ')} already used for this class.`,
        });
        setSaving(false);
        return;
      }

      // update class teacher to the selected teacher
      await supabase.from('classes').update({ teacher_id: teacherId }).eq('id', classId);

      const inserts = slots.map((slot, idx) => {
        const period = periodOptions.find((p) => p.id === slot.periodId)!;
        return {
          name: `${name} (${idx < 2 ? 'Lecture' : 'TDA'})`,
          class_id: classId,
          day_of_week: Number(slot.day),
          start_time: period.start_time,
          end_time: period.end_time,
          start_date: startDate,
          end_date: endDate,
        };
      });

      const { error: insertError } = await supabase.from('subjects').insert(inserts);
      if (insertError) throw insertError;

      toast({ title: 'Subjects created', description: `${name} scheduled for the class.` });
      setDialogOpen(false);
      setName('');
      setClassId('');
      setTeacherId('');
      setStartDate('');
      setEndDate('');
      setSlots([
        { label: 'First Lecture', day: '', periodId: '' },
        { label: 'Second Lecture', day: '', periodId: '' },
        { label: 'First TDA', day: '', periodId: '' },
        { label: 'Second TDA', day: '', periodId: '' },
      ]);
      fetchData();
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error creating subjects',
        description: error.message,
      });
    } finally {
      setSaving(false);
    }
  };

  const renderSlotSelectors = () => (
    <div className="space-y-3">
      {slots.map((slot, idx) => (
        <div key={slot.label} className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label>{slot.label} - Day</Label>
            <Select value={slot.day} onValueChange={(v) => updateSlot(idx, 'day', v)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select day" />
              </SelectTrigger>
              <SelectContent>
                {dayOptions.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>{slot.label} - Period</Label>
            <Select value={slot.periodId} onValueChange={(v) => updateSlot(idx, 'periodId', v)} required>
              <SelectTrigger>
                <SelectValue placeholder="Select period" />
              </SelectTrigger>
              <SelectContent>
                {periodOptions.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      ))}
    </div>
  );

  const handleDeleteGroup = async (group: any) => {
    const ids = group.children.map((c: any) => c.id);
    if (!confirm(`Delete all periods for subject "${group.baseName}"?`)) return;
    try {
      const { error } = await supabase.from('subjects').delete().in('id', ids);
      if (error) throw error;
      setSubjects((prev) => prev.filter((s) => !ids.includes(s.id)));
    } catch (error) {
      console.error('Error deleting subject group:', error);
      toast({
        variant: 'destructive',
        title: 'Error deleting subject',
        description: (error as Error).message,
      });
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete subject "${name}"?`)) return;
    try {
      const { error } = await supabase.from('subjects').delete().eq('id', id);
      if (error) throw error;
      setSubjects((prev) => prev.filter((s) => s.id !== id));
    } catch (error) {
      console.error('Error deleting subject:', error);
      toast({
        variant: 'destructive',
        title: 'Error deleting subject',
        description: (error as Error).message,
      });
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold font-display">Subjects</h1>
            <p className="text-muted-foreground mt-1">
              Manage subjects and schedules across classes.
            </p>
          </div>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="hero">
                <Plus className="w-4 h-4 mr-2" />
                Add Subject
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Subjects</DialogTitle>
                <DialogDescription>
                  Select class, teacher, and day/period for each slot (2 Lectures, 2 TDA). A class cannot repeat a period.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleCreate} className="space-y-4">
                <div className="space-y-2">
                  <Label>Subject Name</Label>
                  <Input
                    placeholder="e.g., Data Structures"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label>Class</Label>
                  <Select value={classId} onValueChange={setClassId} required>
                    <SelectTrigger>
                      <SelectValue placeholder="Select class" />
                    </SelectTrigger>
                    <SelectContent>
                      {classOptions.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value}>
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Teacher</Label>
                  <Select value={teacherId} onValueChange={setTeacherId} required>
                    <SelectTrigger>
                      <SelectValue placeholder="Assign teacher" />
                    </SelectTrigger>
                    <SelectContent>
                      {teacherOptions.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value}>
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label>Course Start Date</Label>
                    <Input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Course End Date</Label>
                    <Input
                      type="date"
                      value={endDate}
                      min={startDate || undefined}
                      onChange={(e) => setEndDate(e.target.value)}
                      required
                    />
                  </div>
                </div>
                {renderSlotSelectors()}
                <Button type="submit" variant="hero" className="w-full" disabled={saving}>
                  {saving ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Create'
                  )}
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" />
              Scheduled Subjects
            </CardTitle>
            <CardDescription>Subjects grouped by class and day.</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">Loading subjects...</div>
            ) : subjects.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">No subjects created yet.</div>
            ) : (
              <div className="space-y-3">
                {Object.values(
                  subjects.reduce((acc: Record<string, any>, subj: any) => {
                    const baseName = subj.name.includes(' (') ? subj.name.split(' (')[0] : subj.name;
                    const key = `${baseName}-${subj.class_id}`;
                    if (!acc[key]) {
                      acc[key] = { baseName, className: subj.classes?.name, teacherId: subj.classes?.teacher_id, children: [] };
                    }
                    acc[key].children.push(subj);
                    return acc;
                  }, {})
                ).map((group: any) => {
                  const first = group.children[0];
                  const isOpen = expandedId === `${group.baseName}-${first.class_id}`;
                  const teacherName =
                    teachers.find((t) => t.user_id === group.teacherId)?.full_name ||
                    teachers.find((t) => t.user_id === group.teacherId)?.email ||
                    'Unknown teacher';
                  return (
                    <button
                      key={`${group.baseName}-${first.class_id}`}
                      type="button"
                      onClick={() =>
                        setExpandedId(isOpen ? null : `${group.baseName}-${first.class_id}`)
                      }
                      className="w-full text-left p-4 rounded-xl border border-border hover:border-primary/20 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">{group.baseName}</p>
                          <p className="text-sm text-muted-foreground">{teacherName}</p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteGroup(group);
                          }}
                        >
                          Delete
                        </Button>
                      </div>
                      {isOpen && (
                        <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                          <p className="text-muted-foreground">
                            Class: {group.className || 'Unknown class'}
                          </p>
                          {group.children.map((child: any) => (
                            <div
                              key={child.id}
                              className="flex flex-col md:flex-row md:items-center md:justify-between gap-2 p-3 rounded-lg border border-border/50"
                            >
                              <div className="flex items-center gap-2">
                                <Badge variant="outline">
                                  {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][child.day_of_week] || child.day_of_week}
                                </Badge>
                                <span>
                                  {child.start_time} - {child.end_time}
                                </span>
                                <span className="ml-2 text-foreground">{child.name}</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toast({
                                      title: 'Edit not implemented',
                                      description: 'Delete the subject and recreate with the desired periods.',
                                    });
                                  }}
                                >
                                  Edit
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
