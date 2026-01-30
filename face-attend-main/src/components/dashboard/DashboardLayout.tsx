import { ReactNode, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
  ClipboardList,
  Sparkles,
  ChevronRight
} from 'lucide-react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '@/lib/auth-context';
import { Button } from '@/components/ui/button';

interface DashboardLayoutProps {
  children: ReactNode;
}

// Navigation links based on role
const studentLinks = [
  { to: '/dashboard', icon: Home, label: 'Dashboard', color: 'from-blue-500 to-cyan-500' },
  { to: '/dashboard/face-register', icon: Camera, label: 'Face Registration', color: 'from-purple-500 to-pink-500' },
  { to: '/dashboard/attendance', icon: CheckCircle, label: 'Mark Attendance', color: 'from-green-500 to-emerald-500' },
  { to: '/dashboard/history', icon: Clock, label: 'My History', color: 'from-orange-500 to-yellow-500' },
];

const teacherLinks = [
  { to: '/dashboard', icon: Home, label: 'Dashboard', color: 'from-blue-500 to-cyan-500' },
  { to: '/dashboard/teacher/attendance', icon: ClipboardList, label: 'View Attendance', color: 'from-purple-500 to-pink-500' },
  { to: '/dashboard/history', icon: Clock, label: 'All History', color: 'from-orange-500 to-yellow-500' },
];

const adminLinks = [
  { to: '/dashboard', icon: Home, label: 'Dashboard', color: 'from-blue-500 to-cyan-500' },
  { to: '/dashboard/admin/students', icon: Users, label: 'Manage Students', color: 'from-purple-500 to-pink-500' },
  { to: '/dashboard/admin/teachers', icon: UserCog, label: 'Manage Teachers', color: 'from-green-500 to-emerald-500' },
  { to: '/dashboard/admin/majors', icon: GraduationCap, label: 'Manage Majors', color: 'from-orange-500 to-yellow-500' },
  { to: '/dashboard/admin/subjects', icon: BookOpen, label: 'Manage Subjects', color: 'from-red-500 to-rose-500' },
  { to: '/dashboard/teacher/attendance', icon: ClipboardList, label: 'View Attendance', color: 'from-cyan-500 to-blue-500' },
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
      {/* Subtle background gradient */}
      <div className="fixed inset-0 -z-10 gradient-mesh opacity-30" />
      
      {/* Mobile overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-20 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ x: sidebarOpen ? 0 : (typeof window !== 'undefined' && window.innerWidth < 768) ? -280 : 0 }}
        className={`fixed md:static z-30 h-screen md:h-auto w-[280px] bg-card/80 backdrop-blur-xl border-r border-border/50 p-5 flex flex-col shadow-xl ${
          sidebarOpen ? '' : '-translate-x-full md:translate-x-0'
        } transition-transform duration-300 ease-out md:transition-none`}
      >
        {/* Logo */}
        <div className="flex items-center justify-between mb-8">
          <Link to="/" className="flex items-center gap-3 group">
            <motion.div 
              className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-500 via-blue-500 to-purple-500 flex items-center justify-center shadow-lg shadow-primary/20"
              whileHover={{ scale: 1.05, rotate: 5 }}
            >
              <Scan className="w-6 h-6 text-white" />
            </motion.div>
            <div>
              <span className="font-display font-bold text-lg text-gradient">FaceAttend</span>
              <span className="block text-[10px] text-muted-foreground uppercase tracking-wider">Dashboard</span>
            </div>
          </Link>
          <motion.button
            className="md:hidden p-2 rounded-xl hover:bg-primary/10 text-muted-foreground"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
            whileTap={{ scale: 0.9 }}
          >
            <X className="w-5 h-5" />
          </motion.button>
        </div>

        {/* AI Badge */}
        <motion.div 
          className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-gradient-to-r from-primary/10 via-purple-500/10 to-pink-500/10 border border-primary/20 mb-6"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="text-sm font-semibold text-gradient">AI Powered</span>
            <span className="block text-[10px] text-muted-foreground">FaceNet + MTCNN</span>
          </div>
        </motion.div>

        {/* Navigation */}
        <nav className="flex-1 space-y-2 overflow-y-auto pr-2 custom-scrollbar">
          {navLinks.map((link, index) => {
            const isActive = location.pathname === link.to;
            return (
              <motion.div
                key={link.to}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Link
                  to={link.to}
                  className={`group flex items-center gap-3 px-4 py-3 rounded-xl transition-all relative overflow-hidden ${
                    isActive
                      ? 'bg-gradient-to-r from-primary/20 to-purple-500/20 text-foreground'
                      : 'text-muted-foreground hover:bg-primary/5 hover:text-foreground'
                  }`}
                  onClick={() => setSidebarOpen(false)}
                >
                  {/* Active indicator */}
                  {isActive && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 rounded-r-full bg-gradient-to-b from-primary to-purple-500"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}
                  
                  {/* Icon with gradient background when active */}
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all ${
                    isActive 
                      ? `bg-gradient-to-br ${link.color} shadow-lg` 
                      : 'bg-muted/50 group-hover:bg-primary/10'
                  }`}>
                    <link.icon className={`w-5 h-5 ${isActive ? 'text-white' : ''}`} />
                  </div>
                  
                  <span className="font-medium flex-1">{link.label}</span>
                  
                  {/* Arrow on hover */}
                  <ChevronRight className={`w-4 h-4 transition-all ${
                    isActive ? 'opacity-100' : 'opacity-0 -translate-x-2 group-hover:opacity-50 group-hover:translate-x-0'
                  }`} />
                </Link>
              </motion.div>
            );
          })}
        </nav>

        {/* User info and Sign out */}
        <div className="pt-4 border-t border-border/50 space-y-3">
          {user && (
            <motion.div 
              className="px-4 py-3 rounded-xl bg-gradient-to-r from-muted/50 to-muted/30 border border-border/50"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <p className="text-sm font-semibold truncate">{user.name}</p>
              <p className="text-xs text-muted-foreground truncate">{user.email}</p>
              <div className="mt-2 flex items-center gap-2">
                <span className={`px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded-full ${
                  user.role === 'admin' 
                    ? 'bg-purple-500/20 text-purple-400' 
                    : user.role === 'teacher'
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-blue-500/20 text-blue-400'
                }`}>
                  {user.role}
                </span>
              </div>
            </motion.div>
          )}
          <Button 
            variant="outline" 
            className="w-full justify-start gap-3 text-muted-foreground hover:text-foreground hover:bg-destructive/10 hover:border-destructive/50 hover:text-destructive transition-all rounded-xl h-11"
            onClick={handleSignOut}
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </Button>
        </div>
      </motion.aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {/* Mobile header */}
        <header className="md:hidden sticky top-0 flex items-center justify-between px-4 py-3 border-b border-border/50 bg-card/80 backdrop-blur-xl z-10">
          <motion.button
            className="p-2.5 rounded-xl hover:bg-primary/10 text-muted-foreground"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sidebar"
            whileTap={{ scale: 0.9 }}
          >
            <Menu className="w-5 h-5" />
          </motion.button>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center">
              <Scan className="w-4 h-4 text-white" />
            </div>
            <span className="font-display font-bold text-gradient">FaceAttend</span>
          </div>
          <div className="w-10" />
        </header>
        
        {/* Page content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
          className="p-4 md:p-8"
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}
