import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Clock, RefreshCw, User, Calendar, ArrowUpDown } from 'lucide-react';
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
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Attendance History</h1>
            <p className="text-muted-foreground mt-1">
              View all recorded attendance entries.
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSortOrder(sortOrder === 'newest' ? 'oldest' : 'newest')}
            >
              <ArrowUpDown className="w-4 h-4 mr-2" />
              {sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}
            </Button>
            <Button variant="outline" size="sm" onClick={fetchRecords} disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-primary" />
              Recent Attendance
            </CardTitle>
            <CardDescription>
              {records.length} records found
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
              </div>
            ) : records.length === 0 ? (
              <div className="text-center py-12">
                <Clock className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No attendance records yet.</p>
                <p className="text-sm text-muted-foreground">
                  Records will appear here after you verify faces.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {sortedRecords.map((record, index) => (
                  <motion.div
                    key={record.id || index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <User className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-semibold">{record.matched_identity}</p>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Calendar className="w-3 h-3" />
                          <span>{formatDate(record.timestamp)}</span>
                          <span>•</span>
                          <Clock className="w-3 h-3" />
                          <span>{formatTime(record.timestamp)}</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">
                        Distance: <span className="text-primary">{record.distance.toFixed(4)}</span>
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {record.distance < 0.3 ? 'High match' : record.distance < 0.5 ? 'Good match' : 'Unknown'}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Stats Card */}
        {records.length > 0 && (
          <div className="grid grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <p className="text-3xl font-bold text-primary">{records.length}</p>
                  <p className="text-sm text-muted-foreground">Total Records</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <p className="text-3xl font-bold text-primary">
                    {new Set(records.map(r => r.matched_identity)).size}
                  </p>
                  <p className="text-sm text-muted-foreground">Unique People</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <p className="text-3xl font-bold text-primary">
                    {(records.reduce((sum, r) => sum + r.distance, 0) / records.length).toFixed(3)}
                  </p>
                  <p className="text-sm text-muted-foreground">Avg Distance</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
