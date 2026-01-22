import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ClipboardList } from 'lucide-react';

export default function ManageAttendance() {
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold font-display">Manage Attendance</h1>
          <p className="text-muted-foreground mt-1">
            Review and update attendance records for your classes.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ClipboardList className="w-5 h-5 text-primary" />
              Attendance Management
            </CardTitle>
            <CardDescription>
              This section will let you review and edit attendance entries.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Coming soon. This page is ready to wire up when attendance editing is implemented.
            </p>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
