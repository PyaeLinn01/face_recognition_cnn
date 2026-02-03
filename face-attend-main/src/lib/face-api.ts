/**
 * MongoDB integration service replacing Supabase.
 * Connects to the backend API for all face recognition operations.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add error handling
apiClient.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// User type
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'student' | 'teacher' | 'admin';
}

// Attendance record type
export interface AttendanceRecord {
  id: string;
  timestamp: string;
  entered_name: string;
  matched_identity: string;
  distance: number;
}

// Admin types
export interface Major {
  id: string;
  name: string;
  description: string;
}

export interface Subject {
  id: string;
  name: string;
  code: string;
}

export interface Teacher {
  id: string;
  name: string;
  email: string;
}

export interface Student {
  id: string;
  name: string;
  email: string;
  face_registered: boolean;
  face_count: number;
}

export interface AttendanceStats {
  total_records: number;
  unique_students: number;
  today_count: number;
}

export const faceAPI = {
  // ==================== AUTH ====================
  
  /**
   * Sign up a new user
   */
  async signup(email: string, password: string, name: string): Promise<{ user: User }> {
    const response = await apiClient.post('/auth/signup', { email, password, name });
    return response.data;
  },

  /**
   * Login user
   */
  async login(email: string, password: string): Promise<{ user: User }> {
    const response = await apiClient.post('/auth/login', { email, password });
    return response.data;
  },

  /**
   * Get user by ID
   */
  async getUser(userId: string): Promise<{ user: User }> {
    const response = await apiClient.get(`/auth/user/${userId}`);
    return response.data;
  },

  // ==================== FACE RECOGNITION ====================
  /**
   * Register a new face for a person
   */
  async registerFace(name: string, imageBase64: string, imageIndex: number) {
    try {
      const response = await apiClient.post('/register-face', {
        name,
        image_base64: imageBase64,
        image_index: imageIndex,
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Verify a face against registered identities
   */
  async verifyFace(imageBase64: string, threshold: number = 0.5, useDetection: boolean = true) {
    try {
      const response = await apiClient.post('/verify-face', {
        image_base64: imageBase64,
        threshold,
        use_detection: useDetection,
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Record attendance for a verified person
   */
  async recordAttendance(name: string, identity: string, distance: number) {
    try {
      const response = await apiClient.post('/attendance/record', {
        name,
        identity,
        distance,
        timestamp: new Date().toISOString(),
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Get recent attendance records
   */
  async getRecentAttendance(limit: number = 50) {
    try {
      const response = await apiClient.get('/attendance/recent', {
        params: { limit },
      });
      return response.data.records;
    } catch (error) {
      throw error;
    }
  },

  /**
   * List all registered faces
   */
  async listRegisteredFaces() {
    try {
      const response = await apiClient.get('/faces/list');
      return response.data.faces;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Detect faces in an image (for real-time overlay)
   * @param identify - Whether to also identify the person (default: true for attendance, false for registration)
   */
  async detectFace(imageBase64: string, minConfidence: number = 0.90, identify: boolean = true, threshold: number = 0.5) {
    try {
      const response = await apiClient.post('/detect-face', {
        image_base64: imageBase64,
        min_confidence: minConfidence,
        identify: identify,
        threshold: threshold,
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Check API health
   */
  async healthCheck() {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Check if a face is real (live) or fake (spoofed)
   * Uses anti-spoofing liveness detection models
   */
  async checkLiveness(imageBase64: string): Promise<{
    is_real: boolean;
    score: number;
    label: 'Real' | 'Fake' | 'No Face';
    face_detected: boolean;
    bbox: [number, number, number, number] | null;
  }> {
    try {
      const response = await apiClient.post('/liveness-check', {
        image_base64: imageBase64,
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // ==================== ADMIN API ====================

  /**
   * Get all majors
   */
  async getMajors(): Promise<Major[]> {
    const response = await apiClient.get('/admin/majors');
    return response.data.majors;
  },

  /**
   * Create a major
   */
  async createMajor(name: string, description: string): Promise<Major> {
    const response = await apiClient.post('/admin/majors', { name, description });
    return response.data.major;
  },

  /**
   * Delete a major
   */
  async deleteMajor(majorId: string): Promise<void> {
    await apiClient.delete(`/admin/majors/${majorId}`);
  },

  /**
   * Get all subjects
   */
  async getSubjects(): Promise<Subject[]> {
    const response = await apiClient.get('/admin/subjects');
    return response.data.subjects;
  },

  /**
   * Create a subject
   */
  async createSubject(name: string, code: string): Promise<Subject> {
    const response = await apiClient.post('/admin/subjects', { name, code });
    return response.data.subject;
  },

  /**
   * Delete a subject
   */
  async deleteSubject(subjectId: string): Promise<void> {
    await apiClient.delete(`/admin/subjects/${subjectId}`);
  },

  /**
   * Get all teachers
   */
  async getTeachers(): Promise<Teacher[]> {
    const response = await apiClient.get('/admin/teachers');
    return response.data.teachers;
  },

  /**
   * Create a teacher
   */
  async createTeacher(name: string, email: string, password: string): Promise<Teacher> {
    const response = await apiClient.post('/admin/teachers', { name, email, password });
    return response.data.teacher;
  },

  /**
   * Delete a teacher
   */
  async deleteTeacher(teacherId: string): Promise<void> {
    await apiClient.delete(`/admin/teachers/${teacherId}`);
  },

  /**
   * Get all students
   */
  async getStudents(): Promise<Student[]> {
    const response = await apiClient.get('/admin/students');
    return response.data.students;
  },

  /**
   * Delete a student
   */
  async deleteStudent(studentId: string): Promise<void> {
    await apiClient.delete(`/admin/students/${studentId}`);
  },

  // ==================== TEACHER API ====================

  /**
   * Get all attendance records (for teachers)
   */
  async getAllAttendance(): Promise<AttendanceRecord[]> {
    const response = await apiClient.get('/teacher/attendance');
    return response.data.records;
  },

  /**
   * Get attendance statistics
   */
  async getAttendanceStats(): Promise<AttendanceStats> {
    const response = await apiClient.get('/teacher/attendance/stats');
    return response.data;
  },
};

export default faceAPI;
