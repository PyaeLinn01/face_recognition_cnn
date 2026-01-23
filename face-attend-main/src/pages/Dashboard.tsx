import { useEffect, useState } from 'react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Camera, CheckCircle, Users, Clock, GraduationCap, BookOpen, UserCog, ClipboardList } from 'lucide-react';
import { Link } from 'react-router-dom';
import { faceAPI } from '@/lib/face-api';
import { useAuth } from '@/lib/auth-context';

interface AttendanceRecord {
  name: string;
  identity: string;
  timestamp: string;
  distance: number;
}

interface RegisteredFace {
  name: string;
  count: number;
}

export default function Dashboard() {
  const { user } = useAuth();
  const [recentAttendance, setRecentAttendance] = useState<AttendanceRecord[]>([]);
  const [registeredFaces, setRegisteredFaces] = useState<RegisteredFace[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking');

  useEffect(() => {
    checkApiHealth();
    fetchData();
  }, []);

  const checkApiHealth = async () => {
    try {
      await faceAPI.healthCheck();
      setApiStatus('online');
    } catch (error) {
      setApiStatus('offline');
    }
  };

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [attendance, faces] = await Promise.all([
        faceAPI.getRecentAttendance(10),
        faceAPI.listRegisteredFaces(),
      ]);
      setRecentAttendance(attendance || []);
      setRegisteredFaces(faces || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Welcome, {user?.name}! ({user?.role})
          </p>
        </div>

        {/* API Status */}
        <Card className={apiStatus === 'online' ? 'border-green-500/50 bg-green-500/5' : apiStatus === 'offline' ? 'border-red-500/50 bg-red-500/5' : ''}>
          <CardContent className="py-4">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${
                apiStatus === 'online' ? 'bg-green-500' : 
                apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
              }`} />
              <span className="font-medium">
                Backend API: {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}
              </span>
              {apiStatus === 'offline' && (
                <span className="text-sm text-muted-foreground">
                  (Make sure python api_server.py is running on port 5001)
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions - Role Based */}
        {user?.role === 'student' && (
          <div className="grid md:grid-cols-2 gap-4">
            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Camera className="w-5 h-5 text-primary" />
                  Register Face
                </CardTitle>
                <CardDescription>
                  Register a new face with 4 photos
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full">
                  <Link to="/dashboard/face-register">
                    Go to Registration
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-primary" />
                  Mark Attendance
                </CardTitle>
                <CardDescription>
                  Verify face and mark attendance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full">
                  <Link to="/dashboard/attendance">
                    Mark Attendance
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {user?.role === 'teacher' && (
          <div className="grid md:grid-cols-2 gap-4">
            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ClipboardList className="w-5 h-5 text-primary" />
                  View Attendance
                </CardTitle>
                <CardDescription>
                  View all student attendance records
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full">
                  <Link to="/dashboard/teacher/attendance">
                    View Records
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-primary" />
                  Attendance History
                </CardTitle>
                <CardDescription>
                  View complete attendance history
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full">
                  <Link to="/dashboard/history">
                    View History
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {user?.role === 'admin' && (
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5 text-primary" />
                  Students
                </CardTitle>
                <CardDescription>Manage student accounts</CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full" size="sm">
                  <Link to="/dashboard/admin/students">Manage</Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <UserCog className="w-5 h-5 text-primary" />
                  Teachers
                </CardTitle>
                <CardDescription>Manage teacher accounts</CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full" size="sm">
                  <Link to="/dashboard/admin/teachers">Manage</Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <GraduationCap className="w-5 h-5 text-primary" />
                  Majors
                </CardTitle>
                <CardDescription>Manage academic majors</CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full" size="sm">
                  <Link to="/dashboard/admin/majors">Manage</Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:border-primary/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-primary" />
                  Subjects
                </CardTitle>
                <CardDescription>Manage course subjects</CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full" size="sm">
                  <Link to="/dashboard/admin/subjects">Manage</Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Stats */}
        <div className="grid md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Users className="w-4 h-4" />
                Registered Faces
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{registeredFaces.length}</div>
              {registeredFaces.length > 0 && (
                <div className="mt-2 space-y-1">
                  {registeredFaces.slice(0, 5).map((face, i) => (
                    <div key={i} className="text-sm text-muted-foreground flex justify-between">
                      <span>{face.name}</span>
                      <span>{face.count} images</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Recent Attendance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{recentAttendance.length}</div>
              {recentAttendance.length > 0 && (
                <div className="mt-2 space-y-1">
                  {recentAttendance.slice(0, 5).map((record, i) => (
                    <div key={i} className="text-sm text-muted-foreground flex justify-between">
                      <span>{record.identity}</span>
                      <span>{new Date(record.timestamp).toLocaleTimeString()}</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}
