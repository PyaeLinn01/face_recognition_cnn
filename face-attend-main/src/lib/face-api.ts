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
}

// Attendance record type
export interface AttendanceRecord {
  id: string;
  timestamp: string;
  entered_name: string;
  matched_identity: string;
  distance: number;
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
};

export default faceAPI;
