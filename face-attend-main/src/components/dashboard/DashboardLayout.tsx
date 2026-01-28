import { ReactNode, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Scan, 
  Home, 
  Camera, 
  CheckCircle,
  Clock,
  LogOut,
  Menu,
  X,
  GraduationCap,
  BookOpen,
  Users,
  UserCog,
  ClipboardList
} from 'lucide-react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '@/lib/auth-context';
import { Button } from '@/components/ui/button';

interface DashboardLayoutProps {
  children: ReactNode;
}

// Navigation links based on role
const studentLinks = [
  { to: '/dashboard', icon: Home, label: 'Dashboard' },
  { to: '/dashboard/face-register', icon: Camera, label: 'Face Registration' },
  { to: '/dashboard/attendance', icon: CheckCircle, label: 'Mark Attendance' },
  { to: '/dashboard/history', icon: Clock, label: 'My History' },
];

const teacherLinks = [
  { to: '/dashboard', icon: Home, label: 'Dashboard' },
  { to: '/dashboard/teacher/attendance', icon: ClipboardList, label: 'View Attendance' },
  { to: '/dashboard/history', icon: Clock, label: 'All History' },
];

const adminLinks = [
  { to: '/dashboard', icon: Home, label: 'Dashboard' },
  { to: '/dashboard/admin/students', icon: Users, label: 'Manage Students' },
  { to: '/dashboard/admin/teachers', icon: UserCog, label: 'Manage Teachers' },
  { to: '/dashboard/admin/majors', icon: GraduationCap, label: 'Manage Majors' },
  { to: '/dashboard/admin/subjects', icon: BookOpen, label: 'Manage Subjects' },
  { to: '/dashboard/teacher/attendance', icon: ClipboardList, label: 'View Attendance' },
];

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const { signOut, user } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
  };

  // Get nav links based on user role
  const getNavLinks = () => {
    if (!user) return studentLinks;
    switch (user.role) {
      case 'admin': return adminLinks;
      case 'teacher': return teacherLinks;
      default: return studentLinks;
    }
  };

  const navLinks = getNavLinks();

  return (
    <div className="min-h-screen bg-background flex">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-20 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed md:static z-30 h-full md:h-auto w-64 bg-card border-r border-border p-4 flex flex-col transform transition-transform duration-200 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        <div className="flex items-center justify-between mb-6">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
              <Scan className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-bold text-xl">FaceAttend</span>
          </Link>
          <button
            className="md:hidden p-2 rounded-lg hover:bg-secondary text-muted-foreground"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Simple info badge */}
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary/10 mb-6">
          <Camera className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-primary">Face Recognition</span>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.to;
            return (
              <Link
                key={link.to}
                to={link.to}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                }`}
                onClick={() => setSidebarOpen(false)}
              >
                <link.icon className="w-5 h-5" />
                <span className="font-medium">{link.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* User info and Sign out */}
        <div className="pt-4 border-t border-border space-y-3">
          {user && (
            <div className="px-3 py-2 rounded-lg bg-secondary/50">
              <p className="text-sm font-medium truncate">{user.name}</p>
              <p className="text-xs text-muted-foreground truncate">{user.email}</p>
            </div>
          )}
          <Button 
            variant="outline" 
            className="w-full justify-start gap-2 text-muted-foreground hover:text-foreground"
            onClick={handleSignOut}
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </Button>
          <p className="text-xs text-muted-foreground text-center">
            MongoDB Backend • FaceNet + MTCNN
          </p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <header className="md:hidden flex items-center justify-between px-4 py-3 border-b border-border bg-card/80 backdrop-blur">
          <button
            className="p-2 rounded-lg hover:bg-secondary text-muted-foreground"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>
          <span className="font-bold">FaceAttend</span>
          <div className="w-10" />
        </header>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="p-4 md:p-8"
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}
