import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Camera, CheckCircle, Users, Clock, GraduationCap, BookOpen, UserCog, ClipboardList, Sparkles, TrendingUp, Activity, Zap } from 'lucide-react';
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

// Animated stat card component
const StatCard = ({ icon: Icon, label, value, color, delay }: {
  icon: typeof Activity;
  label: string;
  value: string | number;
  color: string;
  delay: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5 }}
    whileHover={{ y: -5, transition: { duration: 0.2 } }}
    className="group"
  >
    <Card className="relative overflow-hidden border-border/50 hover:border-primary/30 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5">
      {/* Gradient overlay on hover */}
      <div className={`absolute inset-0 bg-gradient-to-br ${color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`} />
      
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground font-medium">{label}</p>
            <motion.p 
              className="text-3xl font-bold font-display mt-1"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: delay + 0.2, type: 'spring' }}
            >
              {value}
            </motion.p>
          </div>
          <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center shadow-lg`}>
            <Icon className="w-6 h-6 text-white" />
          </div>
        </div>
      </CardContent>
    </Card>
  </motion.div>
);

// Action card component
const ActionCard = ({ icon: Icon, title, description, link, color, delay }: {
  icon: typeof Camera;
  title: string;
  description: string;
  link: string;
  color: string;
  delay: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5 }}
    whileHover={{ y: -5, transition: { duration: 0.2 } }}
    className="group h-full"
  >
    <Card className="relative overflow-hidden border-border/50 hover:border-primary/30 transition-all duration-300 hover:shadow-xl h-full flex flex-col">
      {/* Animated gradient border */}
      <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${color}`} />
      
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <motion.div 
            className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center shadow-lg`}
            whileHover={{ scale: 1.1, rotate: 5 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <Icon className="w-7 h-7 text-white" />
          </motion.div>
          <motion.div
            className="opacity-0 group-hover:opacity-100 transition-opacity"
            initial={{ x: -10 }}
            whileHover={{ x: 0 }}
          >
            <Sparkles className="w-5 h-5 text-primary" />
          </motion.div>
        </div>
        <CardTitle className="text-xl mt-4">{title}</CardTitle>
        <CardDescription className="text-sm">{description}</CardDescription>
      </CardHeader>
      <CardContent className="pt-0 mt-auto">
        <Button asChild variant="hero" className="w-full group/btn">
          <Link to={link} className="flex items-center justify-center gap-2">
            Get Started
            <motion.span
              className="group-hover/btn:translate-x-1 transition-transform"
            >
              →
            </motion.span>
          </Link>
        </Button>
      </CardContent>
    </Card>
  </motion.div>
);

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
      <div className="space-y-8">
        {/* Welcome Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative"
        >
          <div className="absolute -top-4 -left-4 w-32 h-32 bg-gradient-to-br from-primary/20 to-purple-500/20 rounded-full blur-3xl" />
          <h1 className="text-4xl font-bold font-display relative">
            <span className="text-gradient-primary">Dashboard</span>
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Welcome back, <span className="text-foreground font-semibold">{user?.name}</span>! 
            <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary border border-primary/20">
              {user?.role}
            </span>
          </p>
        </motion.div>

        {/* API Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className={`relative overflow-hidden transition-all duration-300 ${
            apiStatus === 'online' 
              ? 'border-emerald-500/30 bg-emerald-500/5 hover:border-emerald-500/50' 
              : apiStatus === 'offline' 
              ? 'border-red-500/30 bg-red-500/5 hover:border-red-500/50' 
              : 'border-amber-500/30 bg-amber-500/5'
          }`}>
            {/* Animated gradient line */}
            <div className={`absolute top-0 left-0 right-0 h-0.5 ${
              apiStatus === 'online' ? 'bg-gradient-to-r from-emerald-500 via-green-400 to-emerald-500' :
              apiStatus === 'offline' ? 'bg-gradient-to-r from-red-500 via-rose-400 to-red-500' :
              'bg-gradient-to-r from-amber-500 via-yellow-400 to-amber-500'
            }`} />
            
            <CardContent className="py-4">
              <div className="flex items-center gap-4">
                <div className={`relative w-3 h-3`}>
                  <div className={`absolute inset-0 rounded-full ${
                    apiStatus === 'online' ? 'bg-emerald-500' : 
                    apiStatus === 'offline' ? 'bg-red-500' : 'bg-amber-500'
                  }`} />
                  <div className={`absolute inset-0 rounded-full animate-ping ${
                    apiStatus === 'online' ? 'bg-emerald-500/50' : 
                    apiStatus === 'offline' ? 'bg-red-500/50' : 'bg-amber-500/50'
                  }`} />
                </div>
                <div className="flex items-center gap-2">
                  <Activity className={`w-4 h-4 ${
                    apiStatus === 'online' ? 'text-emerald-500' : 
                    apiStatus === 'offline' ? 'text-red-500' : 'text-amber-500'
                  }`} />
                  <span className="font-semibold">
                    Backend API: {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}
                  </span>
                </div>
                {apiStatus === 'offline' && (
                  <span className="text-sm text-muted-foreground">
                    (Make sure python api_server.py is running on port 5001)
                  </span>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Quick Actions - Role Based with Stunning Cards */}
        {user?.role === 'student' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              Quick Actions
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <ActionCard
                icon={Camera}
                title="Register Face"
                description="Register a new face with 4 photos for accurate recognition"
                link="/dashboard/face-register"
                color="from-blue-500 to-cyan-500"
                delay={0.3}
              />
              <ActionCard
                icon={CheckCircle}
                title="Mark Attendance"
                description="Verify your face and mark your attendance instantly"
                link="/dashboard/attendance"
                color="from-emerald-500 to-green-500"
                delay={0.4}
              />
            </div>
          </motion.div>
        )}

        {user?.role === 'teacher' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              Quick Actions
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <ActionCard
                icon={ClipboardList}
                title="View Attendance"
                description="View all student attendance records for your classes"
                link="/dashboard/teacher/attendance"
                color="from-violet-500 to-purple-500"
                delay={0.3}
              />
              <ActionCard
                icon={Clock}
                title="Attendance History"
                description="View complete attendance history and analytics"
                link="/dashboard/history"
                color="from-amber-500 to-orange-500"
                delay={0.4}
              />
            </div>
          </motion.div>
        )}

        {user?.role === 'admin' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              Management Panel
            </h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <ActionCard
                icon={Users}
                title="Students"
                description="Manage student accounts and profiles"
                link="/dashboard/admin/students"
                color="from-blue-500 to-cyan-500"
                delay={0.3}
              />
              <ActionCard
                icon={UserCog}
                title="Teachers"
                description="Manage teacher accounts and permissions"
                link="/dashboard/admin/teachers"
                color="from-violet-500 to-purple-500"
                delay={0.35}
              />
              <ActionCard
                icon={GraduationCap}
                title="Majors"
                description="Manage academic majors and programs"
                link="/dashboard/admin/majors"
                color="from-emerald-500 to-green-500"
                delay={0.4}
              />
              <ActionCard
                icon={BookOpen}
                title="Subjects"
                description="Manage course subjects and curriculum"
                link="/dashboard/admin/subjects"
                color="from-amber-500 to-orange-500"
                delay={0.45}
              />
            </div>
          </motion.div>
        )}

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            Overview
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <StatCard
              icon={Users}
              label="Registered Faces"
              value={registeredFaces.length}
              color="from-blue-500 to-cyan-500"
              delay={0.6}
            />
            <StatCard
              icon={Clock}
              label="Recent Attendance"
              value={recentAttendance.length}
              color="from-violet-500 to-purple-500"
              delay={0.7}
            />
          </div>
        </motion.div>

        {/* Detailed Stats Cards */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Registered Faces Detail */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <Card className="relative overflow-hidden border-border/50 hover:border-primary/30 transition-all duration-300 group">
              <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 via-cyan-500 to-blue-500" />
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                    <Users className="w-4 h-4 text-white" />
                  </div>
                  Registered Faces
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-4xl font-bold font-display text-gradient-primary">
                  {registeredFaces.length}
                </div>
                {registeredFaces.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {registeredFaces.slice(0, 5).map((face, i) => (
                      <motion.div 
                        key={i} 
                        className="flex items-center justify-between p-2 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.9 + i * 0.1 }}
                      >
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center text-xs font-bold">
                            {face.name.charAt(0).toUpperCase()}
                          </div>
                          <span className="font-medium">{face.name}</span>
                        </div>
                        <span className="text-sm text-muted-foreground px-2 py-1 rounded-full bg-primary/10">
                          {face.count} images
                        </span>
                      </motion.div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Recent Attendance Detail */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
          >
            <Card className="relative overflow-hidden border-border/50 hover:border-primary/30 transition-all duration-300 group">
              <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-violet-500 via-purple-500 to-violet-500" />
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                    <Clock className="w-4 h-4 text-white" />
                  </div>
                  Recent Attendance
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-4xl font-bold font-display text-gradient-primary">
                  {recentAttendance.length}
                </div>
                {recentAttendance.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {recentAttendance.slice(0, 5).map((record, i) => (
                      <motion.div 
                        key={i} 
                        className="flex items-center justify-between p-2 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 1 + i * 0.1 }}
                      >
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500/20 to-purple-500/20 flex items-center justify-center">
                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                          </div>
                          <span className="font-medium">{record.identity}</span>
                        </div>
                        <span className="text-sm text-muted-foreground px-2 py-1 rounded-full bg-violet-500/10">
                          {new Date(record.timestamp).toLocaleTimeString()}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </DashboardLayout>
  );
}
