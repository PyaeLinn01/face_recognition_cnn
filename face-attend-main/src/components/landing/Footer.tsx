import { Scan } from 'lucide-react';
import { Link } from 'react-router-dom';

export function Footer() {
  return (
    <footer className="py-12 px-4 border-t border-border bg-card">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
              <Scan className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-display font-bold text-xl">FaceAttend</span>
          </Link>
          
          <p className="text-muted-foreground text-sm">
            © {new Date().getFullYear()} FaceAttend. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
