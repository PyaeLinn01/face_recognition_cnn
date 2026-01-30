import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Clock, RefreshCw, User, Calendar, ArrowUpDown, History, Users, Target, CheckCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI, AttendanceRecord } from '@/lib/face-api';

export default function AttendanceHistory() {
  const { toast } = useToast();
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');

  const fetchRecords = async () => {
    setLoading(true);
    try {
      const data = await faceAPI.getRecentAttendance(100);
      setRecords(data || []);
    } catch (error) {
      console.error('Failed to fetch attendance:', error);
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Failed to load attendance records.',
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  const sortedRecords = [...records].sort((a, b) => {
    const dateA = new Date(a.timestamp).getTime();
    const dateB = new Date(b.timestamp).getTime();
    return sortOrder === 'newest' ? dateB - dateA : dateA - dateB;
  });

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Animated Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="relative">
            <div className="absolute -top-4 -left-4 w-32 h-32 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-full blur-3xl" />
            <h1 className="text-4xl font-bold font-display relative">
              <span className="text-gradient-primary">Attendance History</span>
            </h1>
            <p className="text-muted-foreground mt-2 text-lg">
              View all recorded attendance entries
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="heroOutline"
              size="sm"
              onClick={() => setSortOrder(sortOrder === 'newest' ? 'oldest' : 'newest')}
              className="group"
            >
              <ArrowUpDown className="w-4 h-4 mr-2 group-hover:rotate-180 transition-transform duration-300" />
              {sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}
            </Button>
            <Button variant="heroOutline" size="sm" onClick={fetchRecords} disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </motion.div>

        {/* Stats Cards */}
        {records.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-3 gap-4"
          >
            {[
              { 
                icon: History, 
                label: 'Total Records', 
                value: records.length,
                color: 'from-blue-500 to-cyan-500'
              },
              { 
                icon: Users, 
                label: 'Unique People', 
                value: new Set(records.map(r => r.matched_identity)).size,
                color: 'from-violet-500 to-purple-500'
              },
              { 
                icon: Target, 
                label: 'Avg Distance', 
                value: (records.reduce((sum, r) => sum + r.distance, 0) / records.length).toFixed(3),
                color: 'from-emerald-500 to-green-500'
              }
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + i * 0.1 }}
                whileHover={{ y: -3 }}
              >
                <Card className="relative overflow-hidden border-border/50 hover:border-primary/30 transition-all duration-300">
                  <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${stat.color}`} />
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground font-medium">{stat.label}</p>
                        <p className="text-3xl font-bold font-display text-gradient-primary mt-1">{stat.value}</p>
                      </div>
                      <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${stat.color} flex items-center justify-center shadow-lg`}>
                        <stat.icon className="w-6 h-6 text-white" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        )}

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="relative overflow-hidden border-border/50">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-violet-500 via-purple-500 to-pink-500" />
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                  <Clock className="w-5 h-5 text-white" />
                </div>
                Recent Attendance
              </CardTitle>
              <CardDescription className="text-base">
                {records.length} records found
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <motion.div
                    className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-violet-500/20 flex items-center justify-center"
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                  >
                    <RefreshCw className="w-8 h-8 text-primary" />
                  </motion.div>
                </div>
              ) : records.length === 0 ? (
                <div className="text-center py-12">
                  <motion.div
                    className="w-20 h-20 rounded-2xl bg-gradient-to-br from-muted to-muted/50 flex items-center justify-center mx-auto mb-4"
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                  >
                    <Clock className="w-10 h-10 text-muted-foreground" />
                  </motion.div>
                  <p className="text-lg font-medium text-muted-foreground">No attendance records yet</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Records will appear here after you verify faces
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {sortedRecords.map((record, index) => (
                    <motion.div
                      key={record.id || index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.03 }}
                      whileHover={{ x: 4, transition: { duration: 0.2 } }}
                      className="relative flex items-center justify-between p-4 rounded-xl border border-border/50 bg-card hover:border-primary/30 hover:bg-accent/30 transition-all duration-300 group"
                    >
                      {/* Left accent line */}
                      <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-xl ${
                        record.distance < 0.3 ? 'bg-gradient-to-b from-emerald-500 to-green-500' : 
                        record.distance < 0.5 ? 'bg-gradient-to-b from-blue-500 to-cyan-500' : 
                        'bg-gradient-to-b from-amber-500 to-orange-500'
                      }`} />
                      
                      <div className="flex items-center gap-4 pl-2">
                        <motion.div 
                          className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-violet-500/20 flex items-center justify-center group-hover:from-primary/30 group-hover:to-violet-500/30 transition-colors"
                          whileHover={{ scale: 1.05 }}
                        >
                          <User className="w-6 h-6 text-primary" />
                        </motion.div>
                        <div>
                          <p className="font-semibold text-lg">{record.matched_identity}</p>
                          <div className="flex items-center gap-3 text-sm text-muted-foreground">
                            <span className="flex items-center gap-1.5">
                              <Calendar className="w-3.5 h-3.5" />
                              {formatDate(record.timestamp)}
                            </span>
                            <span className="flex items-center gap-1.5">
                              <Clock className="w-3.5 h-3.5" />
                              {formatTime(record.timestamp)}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center gap-2 justify-end">
                          <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                            record.distance < 0.3 ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : 
                            record.distance < 0.5 ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 
                            'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          }`}>
                            {record.distance < 0.3 ? 'High Match' : record.distance < 0.5 ? 'Good Match' : 'Low Match'}
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          Distance: <span className="font-mono font-semibold text-primary">{record.distance.toFixed(4)}</span>
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}
