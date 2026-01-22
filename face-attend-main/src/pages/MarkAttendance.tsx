import { useState, useRef, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Camera, CheckCircle, Loader2, XCircle, RefreshCw } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI } from '@/lib/face-api';

interface DetectedFace {
  box: [number, number, number, number]; // [x, y, width, height]
  confidence: number;
  keypoints?: Record<string, [number, number]>;
  identity?: string | null;  // Identified person name
  match_confidence?: number;  // Match confidence (1 - distance)
}

interface VerificationResult {
  matched: boolean;
  identity: string;
  distance: number;
  threshold: number;
  face_detected?: boolean;
  face_confidence?: number;
}

export default function MarkAttendance() {
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [isMarked, setIsMarked] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Draw face detection boxes on overlay canvas
  const drawFaceBoxes = useCallback((faces: DetectedFace[], videoWidth: number, videoHeight: number) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas size to video
    canvas.width = videoWidth;
    canvas.height = videoHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face) => {
      const [x, y, w, h] = face.box;
      const identity = face.identity;
      const matchConfidence = face.match_confidence || 0;

      // Green for identified, yellow for unknown
      const isIdentified = identity && matchConfidence > 0;
      const boxColor = isIdentified ? '#00ff00' : '#ffcc00';
      const labelBgColor = isIdentified ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 204, 0, 0.9)';

      // Draw rounded rectangle border
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 3;
      ctx.beginPath();
      const radius = 12;
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + w - radius, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
      ctx.lineTo(x + w, y + h - radius);
      ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
      ctx.lineTo(x + radius, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
      ctx.stroke();

      // Create label text - show distance instead of confidence
      let label: string;
      if (isIdentified) {
        // matchConfidence is actually (1 - distance), so distance = 1 - matchConfidence
        const distance = 1 - matchConfidence;
        label = `Welcome, ${identity}! (dist: ${distance.toFixed(3)})`;
      } else {
        label = 'Unknown Face';
      }

      ctx.font = 'bold 14px Arial';
      const textWidth = ctx.measureText(label).width;
      const labelHeight = 28;
      const labelX = x;
      const labelY = y - labelHeight - 4;

      // Background for label
      ctx.fillStyle = labelBgColor;
      ctx.beginPath();
      ctx.roundRect(labelX, Math.max(0, labelY), textWidth + 16, labelHeight, 6);
      ctx.fill();

      // Label text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, labelX + 8, Math.max(17, labelY + 19));

      // Draw keypoints if available
      if (face.keypoints) {
        ctx.fillStyle = '#00ffff';
        Object.values(face.keypoints).forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  }, []);

  // Run face detection periodically
  const runFaceDetection = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isVerifying) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Check if video is ready
    if (video.readyState < 2 || video.videoWidth === 0) return;

    try {
      setIsDetecting(true);

      // Capture current frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.7);
      const imageBase64 = imageDataUrl.split(',')[1];

      // Call detection API
      const result = await faceAPI.detectFace(imageBase64, 0.90);

      if (result.faces && result.faces.length > 0) {
        setDetectedFaces(result.faces);
        drawFaceBoxes(result.faces, video.videoWidth, video.videoHeight);
      } else {
        setDetectedFaces([]);
        // Clear overlay
        const overlayCanvas = overlayCanvasRef.current;
        if (overlayCanvas) {
          const overlayCtx = overlayCanvas.getContext('2d');
          if (overlayCtx) {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
          }
        }
      }
    } catch (error) {
      // Silently fail detection - don't show errors for each frame
      console.debug('Detection error:', error);
    } finally {
      setIsDetecting(false);
    }
  }, [isVerifying, drawFaceBoxes]);

  // Start detection loop when stream is active
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(console.error);

      // Start face detection loop (every 500ms)
      detectionIntervalRef.current = setInterval(() => {
        runFaceDetection();
      }, 500);
    }

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream, runFaceDetection]);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      });
      setStream(mediaStream);
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'Camera Error',
        description: 'Could not access camera. Please check permissions.',
      });
    }
  }, [toast]);

  const stopCamera = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setDetectedFaces([]);
  }, [stream]);

  const reset = useCallback(() => {
    setResult(null);
    setIsMarked(false);
    setDetectedFaces([]);
    startCamera();
  }, [startCamera]);

  const verifyAndMark = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    // Stop detection during verification
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }

    setIsVerifying(true);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not get canvas context');

      ctx.drawImage(video, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      const imageBase64 = imageDataUrl.split(',')[1];

      // Send to backend for verification
      const verifyResult = await faceAPI.verifyFace(imageBase64, 0.7, true);

      setResult({
        matched: verifyResult.matched,
        identity: verifyResult.identity || 'Unknown',
        distance: verifyResult.distance || 999,
        threshold: verifyResult.threshold || 0.7,
        face_detected: verifyResult.face_detected,
        face_confidence: verifyResult.face_confidence,
      });

      if (verifyResult.matched) {
        await faceAPI.recordAttendance(
          verifyResult.identity,
          verifyResult.identity,
          verifyResult.distance
        );

        setIsMarked(true);
        stopCamera();
        toast({
          title: 'Attendance Marked!',
          description: `Welcome, ${verifyResult.identity}! Your attendance has been recorded.`,
        });
      } else {
        const message = verifyResult.face_detected
          ? `Face detected but not recognized. Distance: ${verifyResult.distance?.toFixed(3) || 'N/A'}`
          : 'No face detected. Please ensure your face is visible and well-lit.';
        toast({
          variant: 'destructive',
          title: 'Verification Failed',
          description: message,
        });
        // Restart detection
        detectionIntervalRef.current = setInterval(() => {
          runFaceDetection();
        }, 500);
      }
    } catch (error: any) {
      console.error('Verification error:', error);
      toast({
        variant: 'destructive',
        title: 'Verification Failed',
        description: error?.response?.data?.error || 'Could not verify face. Please try again.',
      });
      // Restart detection
      detectionIntervalRef.current = setInterval(() => {
        runFaceDetection();
      }, 500);
    } finally {
      setIsVerifying(false);
    }
  }, [stopCamera, toast, runFaceDetection]);

  return (
    <DashboardLayout>
      <div className="max-w-2xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Mark Attendance</h1>
          <p className="text-muted-foreground mt-1">
            Use facial recognition to mark your attendance.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="w-5 h-5 text-primary" />
              Face Verification
            </CardTitle>
            <CardDescription>
              Look at the camera to verify your identity.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {isMarked && result?.matched ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center py-12"
              >
                <div className="w-20 h-20 rounded-full bg-green-500 flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Attendance Marked!</h3>
                <p className="text-muted-foreground mb-2">
                  Welcome, <span className="font-semibold text-foreground">{result.identity}</span>
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Distance: <span className="font-mono text-primary">{result.distance.toFixed(4)}</span>
                  <span className="ml-2">(threshold: {result.threshold})</span>
                </p>
                <Button onClick={reset} variant="outline">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Mark Another
                </Button>
              </motion.div>
            ) : result && !result.matched ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center py-12"
              >
                <div className="w-20 h-20 rounded-full bg-red-500 flex items-center justify-center mx-auto mb-4">
                  <XCircle className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-2">
                  {result.face_detected ? 'Face Not Recognized' : 'No Face Detected'}
                </h3>
                <p className="text-muted-foreground mb-2">
                  {result.face_detected
                    ? 'Your face was detected but not matched to any registered identity.'
                    : 'Please ensure your face is visible and well-lit.'}
                </p>
                {result.face_detected && (
                  <p className="text-sm text-muted-foreground mb-4">
                    Distance: {result.distance.toFixed(3)} (threshold: {result.threshold})
                  </p>
                )}
                <div className="flex gap-3 justify-center">
                  <Button onClick={reset} variant="outline">
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Try Again
                  </Button>
                  <Button asChild>
                    <a href="/dashboard/face-register">Register Face</a>
                  </Button>
                </div>
              </motion.div>
            ) : (
              <>
                {/* Video container with overlay canvas */}
                <div className="relative aspect-video bg-black rounded-xl overflow-hidden" style={{ minHeight: '300px' }}>
                  {stream ? (
                    <>
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                        style={{ display: 'block', width: '100%', height: '100%' }}
                      />
                      {/* Overlay canvas for face detection boxes */}
                      <canvas
                        ref={overlayCanvasRef}
                        className="absolute top-0 left-0 w-full h-full pointer-events-none"
                        style={{ objectFit: 'cover' }}
                      />
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center">
                        <Camera className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                        <p className="text-muted-foreground">Camera not started</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Hidden canvas for capture */}
                <canvas ref={canvasRef} className="hidden" />

                {/* Detection status */}
                {stream && (
                  <div className="flex items-center justify-center gap-2 text-sm">
                    <div className={`w-2 h-2 rounded-full ${
                      detectedFaces.length > 0 && detectedFaces[0]?.identity 
                        ? 'bg-green-500' 
                        : detectedFaces.length > 0 
                          ? 'bg-yellow-500' 
                          : 'bg-red-500'
                    } ${isDetecting ? 'animate-pulse' : ''}`} />
                    <span className="text-muted-foreground">
                      {detectedFaces.length > 0 && detectedFaces[0]?.identity
                        ? `Welcome, ${detectedFaces[0].identity}! (dist: ${(1 - (detectedFaces[0]?.match_confidence || 0)).toFixed(3)})`
                        : detectedFaces.length > 0
                          ? 'Face detected - Unknown person'
                          : 'Searching for face...'}
                    </span>
                  </div>
                )}

                <div className="flex gap-3">
                  {!stream ? (
                    <Button className="flex-1" onClick={startCamera}>
                      <Camera className="w-4 h-4 mr-2" />
                      Start Camera
                    </Button>
                  ) : (
                    <Button
                      className="flex-1"
                      onClick={verifyAndMark}
                      disabled={isVerifying || detectedFaces.length === 0}
                    >
                      {isVerifying ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Verifying...
                        </>
                      ) : (
                        <>
                          <CheckCircle className="w-4 h-4 mr-2" />
                          Verify & Mark Attendance
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Instructions</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Make sure your face is registered first</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Wait for the green box to appear around your face</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Ensure good lighting on your face</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Click "Verify & Mark Attendance" when the green box appears</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
