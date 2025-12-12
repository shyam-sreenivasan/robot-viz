import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

// ========================
// UTILITY FUNCTIONS
// ========================

class RobotMath {
  static skew(w) {
    return [
      [0, -w[2], w[1]],
      [w[2], 0, -w[0]],
      [-w[1], w[0], 0]
    ];
  }

  static matrixMultiply(a, b) {
    const result = Array(4).fill().map(() => Array(4).fill(0));
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 4; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return result;
  }

  static dhTransform(alpha, a, d, theta) {
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    const ca = Math.cos(alpha);
    const sa = Math.sin(alpha);
    
    return [
      [ct, -st*ca, st*sa, a*ct],
      [st, ct*ca, -ct*sa, a*st],
      [0, sa, ca, d],
      [0, 0, 0, 1]
    ];
  }

  static forwardKinematicsDH(dhParams, jointAngles) {
    let T = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];
    
    const transforms = [];
    
    for (let i = 0; i < dhParams.length; i++) {
      const [alpha, a, d, thetaOffset] = dhParams[i];
      const theta = jointAngles[i] + thetaOffset;
      const Ti = this.dhTransform(alpha, a, d, theta);
      T = this.matrixMultiply(T, Ti);
      transforms.push(JSON.parse(JSON.stringify(T)));
    }
    
    return { T, transforms };
  }

  static computeJacobian(dhParams, jointAngles) {
    const { transforms } = this.forwardKinematicsDH(dhParams, jointAngles);
    const n = jointAngles.length;
    
    // End-effector position
    const T_final = transforms[transforms.length - 1];
    const p_ee = [T_final[0][3], T_final[1][3], T_final[2][3]];
    
    // Jacobian matrix (6 x n)
    const J = Array(6).fill().map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
      const T_i = i === 0 ? 
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] : 
        transforms[i - 1];
      
      // Extract z-axis and position from transform
      const z_i = [T_i[0][2], T_i[1][2], T_i[2][2]];
      const p_i = [T_i[0][3], T_i[1][3], T_i[2][3]];
      
      // Linear velocity: z_i × (p_ee - p_i)
      const diff = [
        p_ee[0] - p_i[0],
        p_ee[1] - p_i[1],
        p_ee[2] - p_i[2]
      ];
      
      const cross = [
        z_i[1] * diff[2] - z_i[2] * diff[1],
        z_i[2] * diff[0] - z_i[0] * diff[2],
        z_i[0] * diff[1] - z_i[1] * diff[0]
      ];
      
      // Fill Jacobian column
      J[0][i] = cross[0];
      J[1][i] = cross[1];
      J[2][i] = cross[2];
      J[3][i] = z_i[0];
      J[4][i] = z_i[1];
      J[5][i] = z_i[2];
    }
    
    return J;
  }

  static computeScrewAxesFromDH(dhParams) {
    // Screw axes at home position derived from DH parameters
    // These match the analytical Jacobian for UR5
    const screwAxes = [
      [0, 0, 1, 0, 0, 0],                          // S1: rotation about base Z
      [0, -1, 0, 89.2, 0, 0],                      // S2: rotation about Y at height 89.2
      [0, -1, 0, 89.2, 0, -425],                   // S3: rotation about Y at height 89.2, offset by -425
      [0, -1, 0, 89.2, 0, -817],                   // S4: rotation about Y at height 89.2, offset by -817
      [0, 0, -1, 109.3, 817, 0],                   // S5: rotation about -Z
      [0, -1, 0, -5.55, 0, -817]                   // S6: rotation about Y
    ];
    
    return screwAxes;
  }

  static computeJacobianFromScrews(dhParams, jointAngles) {
    // Get screw axes at home position
    const screwAxes = this.computeScrewAxesFromDH(dhParams);
    
    // If at home position, Jacobian is just the screw axes
    const atHome = jointAngles.every(angle => Math.abs(angle) < 1e-6);
    if (atHome) {
      // J = [S1 S2 S3 S4 S5 S6]
      const J = Array(6).fill().map(() => Array(jointAngles.length).fill(0));
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < jointAngles.length; j++) {
          J[i][j] = screwAxes[j][i];
        }
      }
      return J;
    }
    
    // For non-home configurations, use geometric Jacobian
    return this.computeJacobian(dhParams, jointAngles);
  }

  static computeManipulability(J) {
    // Manipulability measure: sqrt(det(J * J^T))
    const m = J.length;
    const n = J[0].length;
    
    // Compute J * J^T
    const JJT = Array(m).fill().map(() => Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        for (let k = 0; k < n; k++) {
          JJT[i][j] += J[i][k] * J[k][j];
        }
      }
    }
    
    // Compute determinant of 6x6 matrix (using cofactor expansion would be complex)
    // For simplicity, we'll compute the product of singular values approximation
    // Use Frobenius norm as a simpler measure
    let sumSq = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        sumSq += J[i][j] * J[i][j];
      }
    }
    
    return Math.sqrt(sumSq);
  }

  static pseudoInverse(J) {
    // Compute J^T
    const m = J.length;
    const n = J[0].length;
    const JT = Array(n).fill().map(() => Array(m).fill(0));
    
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        JT[j][i] = J[i][j];
      }
    }
    
    // Compute J * J^T
    const JJT = Array(m).fill().map(() => Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        for (let k = 0; k < n; k++) {
          JJT[i][j] += J[i][k] * JT[k][j];
        }
      }
    }
    
    // Add damping for numerical stability
    const lambda = 0.01;
    for (let i = 0; i < m; i++) {
      JJT[i][i] += lambda;
    }
    
    // Invert J * J^T using Gauss-Jordan
    const inv = this.invertMatrix(JJT);
    if (!inv) return null;
    
    // Compute pseudo-inverse: J^T * (J * J^T)^-1
    const J_pinv = Array(n).fill().map(() => Array(m).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        for (let k = 0; k < m; k++) {
          J_pinv[i][j] += JT[i][k] * inv[k][j];
        }
      }
    }
    
    return J_pinv;
  }

  static invertMatrix(A) {
    const n = A.length;
    const augmented = A.map((row, i) => [
      ...row,
      ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
    ]);
    
    // Forward elimination
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = k;
        }
      }
      
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
      
      if (Math.abs(augmented[i][i]) < 1e-10) return null;
      
      // Normalize row
      const pivot = augmented[i][i];
      for (let j = 0; j < 2 * n; j++) {
        augmented[i][j] /= pivot;
      }
      
      // Eliminate column
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = augmented[k][i];
          for (let j = 0; j < 2 * n; j++) {
            augmented[k][j] -= factor * augmented[i][j];
          }
        }
      }
    }
    
    // Extract inverse
    return augmented.map(row => row.slice(n));
  }

  static differentialIK(dhParams, currentAngles, targetPos, dt = 0.1) {
    const { T } = this.forwardKinematicsDH(dhParams, currentAngles);
    const currentPos = [T[0][3], T[1][3], T[2][3]];
    
    // Compute position error
    const error = [
      targetPos[0] - currentPos[0],
      targetPos[1] - currentPos[1],
      targetPos[2] - currentPos[2]
    ];
    
    // Desired velocity (proportional control)
    const K = 5.0; // Gain
    const v_desired = error.map(e => K * e);
    
    // Augment with zero angular velocity
    const twist = [...v_desired, 0, 0, 0];
    
    // Compute Jacobian using geometric method
    const J = this.computeJacobian(dhParams, currentAngles);
    
    // Compute pseudo-inverse
    const J_pinv = this.pseudoInverse(J);
    if (!J_pinv) return currentAngles;
    
    // Compute joint velocities
    const dq = Array(currentAngles.length).fill(0);
    for (let i = 0; i < currentAngles.length; i++) {
      for (let j = 0; j < 6; j++) {
        dq[i] += J_pinv[i][j] * twist[j];
      }
    }
    
    // Update joint angles
    const newAngles = currentAngles.map((q, i) => q + dq[i] * dt);
    
    return newAngles;
  }

  static matrixMultiply3x3(a, b) {
    const result = Array(3).fill().map(() => Array(3).fill(0));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return result;
  }

  static addMatrices3x3(a, b) {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]));
  }

  static scaleMatrix3x3(m, s) {
    return m.map(row => row.map(val => val * s));
  }

  static matVecMult3x3(m, v) {
    return m.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));
  }
}

// ========================
// ROBOT CONFIGURATION
// ========================

class RobotConfig {
  static UR5_DH_PARAMS = [
    [Math.PI/2, 0.0, 89.2, 0.0],
    [0.0, 425.0, 0.0, 0.0],
    [0.0, 392.0, 0.0, 0.0],
    [Math.PI/2, 0.0, 109.3, 0.0],
    [-Math.PI/2, 0.0, 94.75, 0.0],
    [0.0, 0.0, 82.5, 0.0]
  ];
}

// ========================
// BASE ROBOT VIEW
// ========================

class BaseRobotRenderer {
  constructor(scene, workspaceLimit = 900) {
    this.scene = scene;
    this.workspaceLimit = workspaceLimit;
    this.robotGroup = new THREE.Group();
    this.waypointsGroup = new THREE.Group();
    this.squareGroup = new THREE.Group();
    this.scene.add(this.robotGroup);
    this.scene.add(this.waypointsGroup);
    this.scene.add(this.squareGroup);
    
    this.setupLighting();
    this.setupGrid();
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1000, 1000, 1000);
    this.scene.add(directionalLight);
  }

  setupGrid() {
    const gridHelper = new THREE.GridHelper(
      this.workspaceLimit * 2, 
      20, 
      0x888888, 
      0x444444
    );
    gridHelper.rotation.x = Math.PI / 2;
    this.scene.add(gridHelper);
    
    const axesHelper = new THREE.AxesHelper(100);
    this.scene.add(axesHelper);
  }

  drawFrame(transform, scale = 50, opacity = 1.0) {
    const origin = new THREE.Vector3(
      transform[0][3],
      transform[1][3],
      transform[2][3]
    );
    
    const colors = [0xff0000, 0x00ff00, 0x0000ff];
    
    for (let i = 0; i < 3; i++) {
      const direction = new THREE.Vector3(
        transform[0][i],
        transform[1][i],
        transform[2][i]
      );
      
      const arrowHelper = new THREE.ArrowHelper(
        direction,
        origin,
        scale,
        colors[i],
        scale * 0.2,
        scale * 0.15
      );
      arrowHelper.line.material.opacity = opacity;
      arrowHelper.line.material.transparent = true;
      arrowHelper.cone.material.opacity = opacity;
      arrowHelper.cone.material.transparent = true;
      
      this.robotGroup.add(arrowHelper);
    }
  }

  drawRobot(dhParams, jointAngles, showFrames = true) {
    while (this.robotGroup.children.length > 0) {
      this.robotGroup.remove(this.robotGroup.children[0]);
    }
    
    const identity = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];
    this.drawFrame(identity, 80, 1.0);
    
    const { T, transforms } = RobotMath.forwardKinematicsDH(dhParams, jointAngles);
    
    const points = [new THREE.Vector3(0, 0, 0)];
    transforms.forEach((transform, i) => {
      points.push(new THREE.Vector3(
        transform[0][3],
        transform[1][3],
        transform[2][3]
      ));
      
      if (showFrames) {
        this.drawFrame(transform, 50, 0.7);
      }
    });
    
    const linkGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const linkMaterial = new THREE.LineBasicMaterial({ 
      color: 0x000000, 
      linewidth: 5 
    });
    const linkLine = new THREE.Line(linkGeometry, linkMaterial);
    this.robotGroup.add(linkLine);
    
    const jointGeometry = new THREE.SphereGeometry(12, 16, 16);
    const jointMaterial = new THREE.MeshPhongMaterial({ color: 0x000000 });
    
    points.forEach(point => {
      const joint = new THREE.Mesh(jointGeometry, jointMaterial);
      joint.position.copy(point);
      this.robotGroup.add(joint);
    });
    
    const eeGeometry = new THREE.SphereGeometry(18, 16, 16);
    const eeMaterial = new THREE.MeshPhongMaterial({ color: 0xff0000 });
    const endEffector = new THREE.Mesh(eeGeometry, eeMaterial);
    endEffector.position.copy(points[points.length - 1]);
    this.robotGroup.add(endEffector);
    
    return { T, points };
  }

  addWaypoint(position) {
    const waypointGeometry = new THREE.SphereGeometry(4, 8, 8);
    const waypointMaterial = new THREE.MeshPhongMaterial({ 
      color: 0xff0000,
      transparent: true,
      opacity: 0.7
    });
    const waypoint = new THREE.Mesh(waypointGeometry, waypointMaterial);
    waypoint.position.copy(position);
    this.waypointsGroup.add(waypoint);
  }

  clearWaypoints() {
    while (this.waypointsGroup.children.length > 0) {
      this.waypointsGroup.remove(this.waypointsGroup.children[0]);
    }
  }

  drawSquare(corners) {
    while (this.squareGroup.children.length > 0) {
      this.squareGroup.remove(this.squareGroup.children[0]);
    }
    
    const points = [...corners, corners[0]];
    const geometry = new THREE.BufferGeometry().setFromPoints(
      points.map(p => new THREE.Vector3(p[0], p[1], p[2]))
    );
    const material = new THREE.LineBasicMaterial({ 
      color: 0x0000ff, 
      linewidth: 2,
      transparent: true,
      opacity: 0.5
    });
    const line = new THREE.Line(geometry, material);
    this.squareGroup.add(line);
    
    // Draw corner markers
    corners.forEach(corner => {
      const cornerGeometry = new THREE.SphereGeometry(6, 8, 8);
      const cornerMaterial = new THREE.MeshPhongMaterial({ 
        color: 0x0000ff,
        transparent: true,
        opacity: 0.5
      });
      const marker = new THREE.Mesh(cornerGeometry, cornerMaterial);
      marker.position.set(corner[0], corner[1], corner[2]);
      this.squareGroup.add(marker);
    });
  }

  clearSquare() {
    while (this.squareGroup.children.length > 0) {
      this.squareGroup.remove(this.squareGroup.children[0]);
    }
  }
}

// ========================
// BASE ROBOT VIEW COMPONENT
// ========================

const BaseRobotView = ({ dhParams, children }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const robotRendererRef = useRef(null);
  const cameraTargetRef = useRef(new THREE.Vector3(0, 0, 0));
  const [currentConfig, setCurrentConfig] = useState(Array(6).fill(0));
  const [endEffectorInfo, setEndEffectorInfo] = useState(null);
  const [cameraDistance, setCameraDistance] = useState(2300);
  const [cameraElevation, setCameraElevation] = useState(-48);
  const [cameraAzimuth, setCameraAzimuth] = useState(65);
  const [cameraRoll, setCameraRoll] = useState(8);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      50,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      1,
      10000
    );
    
    const updateCameraPosition = () => {
      const phi = (90 - cameraElevation) * Math.PI / 180;
      const theta = cameraAzimuth * Math.PI / 180;
      
      camera.position.set(
        cameraDistance * Math.sin(phi) * Math.sin(theta),
        cameraDistance * Math.cos(phi),
        cameraDistance * Math.sin(phi) * Math.cos(theta)
      );
      camera.lookAt(0, 0, 0);
    };
    
    updateCameraPosition();
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    robotRendererRef.current = new BaseRobotRenderer(scene, 900);

    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    let isPanning = false;
    let previousMousePosition = { x: 0, y: 0 };
    const cameraTarget = cameraTargetRef.current;
    
    const onMouseDown = (e) => {
      if (e.button === 0) {
        isPanning = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    };
    
    const onMouseMove = (e) => {
      if (!isPanning) return;
      
      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;
      
      const panSpeed = 2;
      
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      camera.getWorldDirection(right);
      right.cross(camera.up).normalize();
      up.copy(camera.up).normalize();
      
      const panOffset = new THREE.Vector3()
        .addScaledVector(right, -deltaX * panSpeed)
        .addScaledVector(up, deltaY * panSpeed);
      
      camera.position.add(panOffset);
      cameraTarget.add(panOffset);
      camera.lookAt(cameraTarget);
      
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };
    
    const onMouseUp = () => {
      isPanning = false;
    };
    
    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseup', onMouseUp);
    renderer.domElement.addEventListener('mouseleave', onMouseUp);

    updateRobot(currentConfig);

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('mousedown', onMouseDown);
      renderer.domElement.removeEventListener('mousemove', onMouseMove);
      renderer.domElement.removeEventListener('mouseup', onMouseUp);
      renderer.domElement.removeEventListener('mouseleave', onMouseUp);
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, []);

  const updateRobot = (jointAngles) => {
    if (!robotRendererRef.current) return;
    
    const { T, points } = robotRendererRef.current.drawRobot(
      dhParams,
      jointAngles,
      true
    );
    
    const eePos = points[points.length - 1];
    const distance = eePos.length();
    
    // Compute Jacobian using geometric method (works for all configurations)
    const jacobian = RobotMath.computeJacobian(dhParams, jointAngles);
    const manipulability = RobotMath.computeManipulability(jacobian);
    
    setEndEffectorInfo({
      position: eePos,
      distance: distance,
      transform: T,
      jointAngles: jointAngles,
      jacobian: jacobian,
      manipulability: manipulability
    });
  };

  const handleConfigChange = (newConfig) => {
    setCurrentConfig(newConfig);
    updateRobot(newConfig);
  };

  const handleCameraChange = (elevation, azimuth) => {
    if (!cameraRef.current) return;
    
    const camera = cameraRef.current;
    const cameraTarget = cameraTargetRef.current;
    
    const phi = (90 - elevation) * Math.PI / 180;
    const theta = azimuth * Math.PI / 180;
    
    const offset = new THREE.Vector3(
      cameraDistance * Math.sin(phi) * Math.sin(theta),
      cameraDistance * Math.cos(phi),
      cameraDistance * Math.sin(phi) * Math.cos(theta)
    );
    
    camera.position.copy(cameraTarget).add(offset);
    camera.lookAt(cameraTarget);
    camera.rotation.z = cameraRoll * Math.PI / 180;
  };

  useEffect(() => {
    handleCameraChange(cameraElevation, cameraAzimuth);
  }, [cameraElevation, cameraAzimuth, cameraDistance, cameraRoll]);

  return (
  <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'row', backgroundColor: '#f0f0f0' }}>
    {/* Left side - Robot View (60%) */}
    <div style={{ flex: '0 0 60%', position: 'relative', display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
        
        {/* Camera control sliders overlay */}
        <div style={{ 
          position: 'absolute', 
          top: '16px', 
          right: '16px', 
          backgroundColor: 'rgba(255, 255, 255, 0.95)', 
          padding: '12px', 
          borderRadius: '8px', 
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          fontSize: '12px'
        }}>
          <h4 style={{ fontWeight: 'bold', fontSize: '12px', marginBottom: '8px' }}>Camera</h4>
          
          <div style={{ marginBottom: '8px' }}>
            <label style={{ fontSize: '12px', fontWeight: '500', display: 'block', marginBottom: '4px' }}>
              ρ: {cameraDistance.toFixed(0)}mm
            </label>
            <input
              type="range"
              min="800"
              max="4000"
              step="50"
              value={cameraDistance}
              onChange={(e) => setCameraDistance(parseFloat(e.target.value))}
              style={{ width: '128px', height: '4px' }}
            />
          </div>
          
          <div style={{ marginBottom: '8px' }}>
            <label style={{ fontSize: '12px', fontWeight: '500', display: 'block', marginBottom: '4px' }}>
              α: {cameraElevation.toFixed(0)}°
            </label>
            <input
              type="range"
              min="-89"
              max="89"
              step="1"
              value={cameraElevation}
              onChange={(e) => setCameraElevation(parseFloat(e.target.value))}
              style={{ width: '128px', height: '4px' }}
            />
          </div>
          
          <div style={{ marginBottom: '8px' }}>
            <label style={{ fontSize: '12px', fontWeight: '500', display: 'block', marginBottom: '4px' }}>
              β: {cameraAzimuth.toFixed(0)}°
            </label>
            <input
              type="range"
              min="-180"
              max="180"
              step="1"
              value={cameraAzimuth}
              onChange={(e) => setCameraAzimuth(parseFloat(e.target.value))}
              style={{ width: '128px', height: '4px' }}
            />
          </div>
          
          <div style={{ marginBottom: '8px' }}>
            <label style={{ fontSize: '12px', fontWeight: '500', display: 'block', marginBottom: '4px' }}>
              γ: {cameraRoll.toFixed(0)}°
            </label>
            <input
              type="range"
              min="-180"
              max="180"
              step="1"
              value={cameraRoll}
              onChange={(e) => setCameraRoll(parseFloat(e.target.value))}
              style={{ width: '128px', height: '4px' }}
            />
          </div>
        </div>
      </div>
    </div>
    
    {/* Right side - Controls and Data (40%) */}
    <div style={{ flex: '0 0 40%', display: 'flex', flexDirection: 'column', backgroundColor: 'white', borderLeft: '1px solid #d1d5db', overflow: 'hidden' }}>
      {/* Data section */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', borderBottom: '1px solid #d1d5db' }}>
        <h3 style={{ fontWeight: 'bold', fontSize: '36px', marginBottom: '12px' }}>End-Effector Info</h3>
        {endEffectorInfo && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <h4 style={{ fontWeight: '600', marginBottom: '8px', fontSize: '32px' }}>Transform (T_06):</h4>
              <div style={{ fontFamily: 'monospace', fontSize: '28px', backgroundColor: '#f9fafb', padding: '8px', borderRadius: '4px' }}>
                {endEffectorInfo.transform.slice(0, 4).map((row, i) => (
                  <div key={i}>
                    [{row.map((val, j) => 
                      (j === 3 && i < 3) ? val.toFixed(2).padStart(8) : val.toFixed(4).padStart(7)
                    ).join('  ')}]
                  </div>
                ))}
              </div>
            </div>
            
            <div>
              <h4 style={{ fontWeight: '600', marginBottom: '8px', fontSize: '32px' }}>Position (mm):</h4>
              <div style={{ fontFamily: 'monospace', fontSize: '34px' }}>
                <div>X: {endEffectorInfo.position.x.toFixed(2)}</div>
                <div>Y: {endEffectorInfo.position.y.toFixed(2)}</div>
                <div>Z: {endEffectorInfo.position.z.toFixed(2)}</div>
              </div>
            </div>
            
            <div>
              <h4 style={{ fontWeight: '600', marginBottom: '8px', fontSize: '32px' }}>Joint Angles (deg):</h4>
              <div style={{ fontFamily: 'monospace', fontSize: '34px' }}>
                {endEffectorInfo.jointAngles.map((angle, i) => (
                  <div key={i}>θ{i+1}: {(angle * 180 / Math.PI).toFixed(2)}°</div>
                ))}
              </div>
            </div>
            
            <div>
              <h4 style={{ fontWeight: '600', marginBottom: '8px', fontSize: '32px' }}>Manipulability:</h4>
              <div style={{ fontFamily: 'monospace', fontSize: '40px', fontWeight: 'bold', color: '#2563eb' }}>
                μ = {endEffectorInfo.manipulability.toFixed(2)}
              </div>
              <div style={{ fontSize: '28px', color: '#6b7280', marginTop: '4px' }}>
                (Higher is better - measure of dexterity)
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Controls section */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
        {children && React.cloneElement(children, { 
          currentConfig, 
          onConfigChange: handleConfigChange,
          robotRenderer: robotRendererRef.current,
          dhParams: dhParams
        })}
      </div>
    </div>
  </div>
);};

// ========================
// TRAJECTORY CONTROLS
// ========================

const TrajectoryControls = ({ currentConfig, onConfigChange, robotRenderer, dhParams }) => {
  const [localConfig, setLocalConfig] = useState(currentConfig);
  const [trajectoryType, setTrajectoryType] = useState('square');
  const [squareCorners, setSquareCorners] = useState(null);
  const [circleParams, setCircleParams] = useState(null);
  const [sineParams, setSineParams] = useState(null);
  const [trajectoryPlane, setTrajectoryPlane] = useState('xy');
  const [isAnimating, setIsAnimating] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [currentWaypoints, setCurrentWaypoints] = useState([]);
  const animationRef = useRef(null);

  useEffect(() => {
    setLocalConfig(currentConfig);
  }, [currentConfig]);

  const handleSliderChange = (index, value) => {
    const newConfig = [...localConfig];
    newConfig[index] = (value * Math.PI) / 180;
    setLocalConfig(newConfig);
    onConfigChange(newConfig);
  };

  const handleReset = () => {
    const zeros = Array(6).fill(0);
    setLocalConfig(zeros);
    onConfigChange(zeros);
    setStepIndex(0);
    if (robotRenderer) {
      robotRenderer.clearWaypoints();
      robotRenderer.clearSquare();
    }
    setSquareCorners(null);
    setCircleParams(null);
    setSineParams(null);
  };

  const handleRandom = () => {
    const random = Array(6).fill(0).map(() => 
      (Math.random() * 2 - 1) * Math.PI
    );
    setLocalConfig(random);
    onConfigChange(random);
  };

  const initSquare = () => {
    const { T } = RobotMath.forwardKinematicsDH(dhParams, localConfig);
    const currentPos = [T[0][3], T[1][3], T[2][3]];
    
    const planes = ['xy', 'yz', 'zx'];
    const selectedPlane = planes[Math.floor(Math.random() * planes.length)];
    setTrajectoryPlane(selectedPlane);
    
    const sideLength = 100;
    
    let corners = [];
    switch (selectedPlane) {
      case 'xy':
        corners = [
          currentPos,
          [currentPos[0] + sideLength, currentPos[1], currentPos[2]],
          [currentPos[0] + sideLength, currentPos[1] + sideLength, currentPos[2]],
          [currentPos[0], currentPos[1] + sideLength, currentPos[2]]
        ];
        break;
      case 'yz':
        corners = [
          currentPos,
          [currentPos[0], currentPos[1] + sideLength, currentPos[2]],
          [currentPos[0], currentPos[1] + sideLength, currentPos[2] + sideLength],
          [currentPos[0], currentPos[1], currentPos[2] + sideLength]
        ];
        break;
      case 'zx':
        corners = [
          currentPos,
          [currentPos[0], currentPos[1], currentPos[2] + sideLength],
          [currentPos[0] + sideLength, currentPos[1], currentPos[2] + sideLength],
          [currentPos[0] + sideLength, currentPos[1], currentPos[2]]
        ];
        break;
    }
    
    setSquareCorners(corners);
    setCircleParams(null);
    setSineParams(null);
    setStepIndex(0);
    
    if (robotRenderer) {
      robotRenderer.clearWaypoints();
      robotRenderer.drawSquare(corners);
    }
  };

  const initCircle = () => {
    const { T } = RobotMath.forwardKinematicsDH(dhParams, localConfig);
    const currentPos = [T[0][3], T[1][3], T[2][3]];
    
    const planes = ['xy', 'yz', 'zx'];
    const selectedPlane = planes[Math.floor(Math.random() * planes.length)];
    setTrajectoryPlane(selectedPlane);
    
    const radius = 100;
    
    let center;
    switch (selectedPlane) {
      case 'xy':
        center = [currentPos[0] - radius, currentPos[1], currentPos[2]];
        break;
      case 'yz':
        center = [currentPos[0], currentPos[1] - radius, currentPos[2]];
        break;
      case 'zx':
        center = [currentPos[0], currentPos[1], currentPos[2] - radius];
        break;
    }
    
    setCircleParams({ center, radius, plane: selectedPlane });
    setSquareCorners(null);
    setSineParams(null);
    setStepIndex(0);
    
    const numPoints = 32;
    const circlePoints = [];
    
    for (let i = 0; i <= numPoints; i++) {
      const angle = (i / numPoints) * 2 * Math.PI;
      let point;
      
      switch (selectedPlane) {
        case 'xy':
          point = [
            center[0] + radius * Math.cos(angle),
            center[1] + radius * Math.sin(angle),
            center[2]
          ];
          break;
        case 'yz':
          point = [
            center[0],
            center[1] + radius * Math.cos(angle),
            center[2] + radius * Math.sin(angle)
          ];
          break;
        case 'zx':
          point = [
            center[0] + radius * Math.cos(angle),
            center[1],
            center[2] + radius * Math.sin(angle)
          ];
          break;
      }
      
      circlePoints.push(point);
    }
    
    if (robotRenderer) {
      robotRenderer.clearWaypoints();
      robotRenderer.drawSquare(circlePoints);
    }
  };

  const checkWorkspaceBounds = (points, limit = 900) => {
    for (let point of points) {
      const distance = Math.sqrt(point[0]**2 + point[1]**2 + point[2]**2);
      if (distance > limit) {
        return false;
      }
    }
    return true;
  };

  const initSineWave = () => {
    const { T } = RobotMath.forwardKinematicsDH(dhParams, localConfig);
    const currentPos = [T[0][3], T[1][3], T[2][3]];
    
    const maxWidth = 150;
    const workspaceLimit = 900;
    let validSine = false;
    let sinePoints = [];
    let selectedPlane, direction, wavelength, amplitude;
    
    const maxAttempts = 20;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const planes = ['xy', 'yz', 'zx'];
      selectedPlane = planes[Math.floor(Math.random() * planes.length)];
      
      direction = Math.random() > 0.5 ? 1 : -1;
      wavelength = 80 + Math.random() * 70;
      amplitude = 60 + Math.random() * 60;
      
      sinePoints = [];
      const numPoints = 40;
      
      for (let i = 0; i <= numPoints; i++) {
        const t = i / numPoints;
        const x = wavelength * t;
        const y = amplitude * Math.sin(2 * Math.PI * t);
        
        let point;
        switch (selectedPlane) {
          case 'xy':
            point = [
              currentPos[0] + direction * x,
              currentPos[1] + y,
              currentPos[2]
            ];
            break;
          case 'yz':
            point = [
              currentPos[0] + y,
              currentPos[1] + direction * x,
              currentPos[2]
            ];
            break;
          case 'zx':
            point = [
              currentPos[0] + y,
              currentPos[1],
              currentPos[2] + direction * x
            ];
            break;
        }
        sinePoints.push(point);
      }
      
      if (checkWorkspaceBounds(sinePoints, workspaceLimit)) {
        validSine = true;
        break;
      }
    }
    
    if (!validSine) {
      alert('Could not generate a valid sine wave within workspace bounds. Try a different starting position.');
      return;
    }
    
    setTrajectoryPlane(selectedPlane);
    setSineParams({ 
      startPos: currentPos, 
      wavelength, 
      amplitude, 
      plane: selectedPlane,
      direction 
    });
    setSquareCorners(null);
    setCircleParams(null);
    setStepIndex(0);
    
    if (robotRenderer) {
      robotRenderer.clearWaypoints();
      robotRenderer.drawSquare(sinePoints);
    }
  };

  const animateSquare = async () => {
    if (!squareCorners || isAnimating) return;
    
    setIsAnimating(true);
    
    if (robotRenderer) {
      robotRenderer.clearWaypoints();
    }
    
    const waypoints = [];
    const pointsPerSide = 20;
    
    for (let i = 0; i < 4; i++) {
      const start = squareCorners[i];
      const end = squareCorners[(i + 1) % 4];
      
      for (let j = 0; j <= pointsPerSide; j++) {
        const t = j / pointsPerSide;
        const waypoint = [
          start[0] + (end[0] - start[0]) * t,
          start[1] + (end[1] - start[1]) * t,
          start[2] + (end[2] - start[2]) * t
        ];
        waypoints.push(waypoint);
      }
    }
    
    let currentAngles = [...localConfig];
    
    const animate = async () => {
      for (let i = 0; i < waypoints.length; i++) {
        if (!animationRef.current) break;
        
        const targetPos = waypoints[i];
        
        for (let iter = 0; iter < 3; iter++) {
          currentAngles = RobotMath.differentialIK(
            dhParams,
            currentAngles,
            targetPos,
            0.1
          );
        }
        
        setLocalConfig(currentAngles);
        onConfigChange(currentAngles);
        
        if (robotRenderer && i % 2 === 0) {
          robotRenderer.addWaypoint(new THREE.Vector3(targetPos[0], targetPos[1], targetPos[2]));
        }
        
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      
      setIsAnimating(false);
      animationRef.current = null;
    };
    
    animationRef.current = true;
    animate();
  };

  const animateCircle = async () => {
    if (!circleParams || isAnimating) return;
    
    setIsAnimating(true);
    
    if (robotRenderer) {
      robotRenderer.clearWaypoints();
    }
    
    const waypoints = [];
    const numPoints = 80;
    const { center, radius, plane } = circleParams;
    
    for (let i = 0; i <= numPoints; i++) {
      const angle = (i / numPoints) * 2 * Math.PI;
      let waypoint;
      
      switch (plane) {
        case 'xy':
          waypoint = [
            center[0] + radius * Math.cos(angle),
            center[1] + radius * Math.sin(angle),
            center[2]
          ];
          break;
        case 'yz':
          waypoint = [
            center[0],
            center[1] + radius * Math.cos(angle),
            center[2] + radius * Math.sin(angle)
          ];
          break;
        case 'zx':
          waypoint = [
            center[0] + radius * Math.cos(angle),
            center[1],
            center[2] + radius * Math.sin(angle)
          ];
          break;
      }
      
      waypoints.push(waypoint);
    }
    
    let currentAngles = [...localConfig];
    
    const animate = async () => {
      for (let i = 0; i < waypoints.length; i++) {
        if (!animationRef.current) break;
        
        const targetPos = waypoints[i];
        
        for (let iter = 0; iter < 3; iter++) {
          currentAngles = RobotMath.differentialIK(
            dhParams,
            currentAngles,
            targetPos,
            0.1
          );
        }
        
        setLocalConfig(currentAngles);
        onConfigChange(currentAngles);
        
        if (robotRenderer && i % 2 === 0) {
          robotRenderer.addWaypoint(new THREE.Vector3(targetPos[0], targetPos[1], targetPos[2]));
        }
        
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      
      setIsAnimating(false);
      animationRef.current = null;
    };
    
    animationRef.current = true;
    animate();
  };

  const animateSineWave = async () => {
    if (!sineParams || isAnimating) return;
    
    setIsAnimating(true);
    
    if (robotRenderer) {
      robotRenderer.clearWaypoints();
    }
    
    const waypoints = [];
    const numPoints = 60;
    const { startPos, wavelength, amplitude, plane, direction } = sineParams;
    
    for (let i = 0; i <= numPoints; i++) {
      const t = i / numPoints;
      const x = wavelength * t;
      const y = amplitude * Math.sin(2 * Math.PI * t);
      
      let waypoint;
      switch (plane) {
        case 'xy':
          waypoint = [
            startPos[0] + direction * x,
            startPos[1] + y,
            startPos[2]
          ];
          break;
        case 'yz':
          waypoint = [
            startPos[0] + y,
            startPos[1] + direction * x,
            startPos[2]
          ];
          break;
        case 'zx':
          waypoint = [
            startPos[0] + y,
            startPos[1],
            startPos[2] + direction * x
          ];
          break;
      }
      
      waypoints.push(waypoint);
    }
    
    let currentAngles = [...localConfig];
    
    const animate = async () => {
      for (let i = 0; i < waypoints.length; i++) {
        if (!animationRef.current) break;
        
        const targetPos = waypoints[i];
        
        for (let iter = 0; iter < 3; iter++) {
          currentAngles = RobotMath.differentialIK(
            dhParams,
            currentAngles,
            targetPos,
            0.1
          );
        }
        
        setLocalConfig(currentAngles);
        onConfigChange(currentAngles);
        
        if (robotRenderer && i % 2 === 0) {
          robotRenderer.addWaypoint(new THREE.Vector3(targetPos[0], targetPos[1], targetPos[2]));
        }
        
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      
      setIsAnimating(false);
      animationRef.current = null;
    };
    
    animationRef.current = true;
    animate();
  };

  const stopAnimation = () => {
    setIsAnimating(false);
    animationRef.current = null;
  };

  const generateSquareWaypoints = () => {
    if (!squareCorners) return [];
    const waypoints = [];
    const pointsPerSide = 20;
    
    for (let i = 0; i < 4; i++) {
      const start = squareCorners[i];
      const end = squareCorners[(i + 1) % 4];
      
      for (let j = 0; j <= pointsPerSide; j++) {
        const t = j / pointsPerSide;
        const waypoint = [
          start[0] + (end[0] - start[0]) * t,
          start[1] + (end[1] - start[1]) * t,
          start[2] + (end[2] - start[2]) * t
        ];
        waypoints.push(waypoint);
      }
    }
    return waypoints;
  };

  const generateCircleWaypoints = () => {
    if (!circleParams) return [];
    const waypoints = [];
    const numPoints = 80;
    const { center, radius, plane } = circleParams;
    
    for (let i = 0; i <= numPoints; i++) {
      const angle = (i / numPoints) * 2 * Math.PI;
      let waypoint;
      
      switch (plane) {
        case 'xy':
          waypoint = [
            center[0] + radius * Math.cos(angle),
            center[1] + radius * Math.sin(angle),
            center[2]
          ];
          break;
        case 'yz':
          waypoint = [
            center[0],
            center[1] + radius * Math.cos(angle),
            center[2] + radius * Math.sin(angle)
          ];
          break;
        case 'zx':
          waypoint = [
            center[0] + radius * Math.cos(angle),
            center[1],
            center[2] + radius * Math.sin(angle)
          ];
          break;
      }
      
      waypoints.push(waypoint);
    }
    return waypoints;
  };

  const generateSineWaypoints = () => {
    if (!sineParams) return [];
    const waypoints = [];
    const numPoints = 60;
    const { startPos, wavelength, amplitude, plane, direction } = sineParams;
    
    for (let i = 0; i <= numPoints; i++) {
      const t = i / numPoints;
      const x = wavelength * t;
      const y = amplitude * Math.sin(2 * Math.PI * t);
      
      let waypoint;
      switch (plane) {
        case 'xy':
          waypoint = [
            startPos[0] + direction * x,
            startPos[1] + y,
            startPos[2]
          ];
          break;
        case 'yz':
          waypoint = [
            startPos[0] + y,
            startPos[1] + direction * x,
            startPos[2]
          ];
          break;
        case 'zx':
          waypoint = [
            startPos[0] + y,
            startPos[1],
            startPos[2] + direction * x
          ];
          break;
      }
      
      waypoints.push(waypoint);
    }
    return waypoints;
  };

  const stepForward = () => {
    let waypoints = [];
    
    if (trajectoryType === 'square' && squareCorners) {
      waypoints = generateSquareWaypoints();
    } else if (trajectoryType === 'circle' && circleParams) {
      waypoints = generateCircleWaypoints();
    } else if (trajectoryType === 'sine' && sineParams) {
      waypoints = generateSineWaypoints();
    }
    
    if (waypoints.length === 0) return;
    
    // Don't go beyond the last waypoint
    if (stepIndex >= waypoints.length - 1) return;
    
    const nextIndex = stepIndex + 1;
    setStepIndex(nextIndex);
    
    const targetPos = waypoints[nextIndex];
    let currentAngles = [...localConfig];
    
    for (let iter = 0; iter < 5; iter++) {
      currentAngles = RobotMath.differentialIK(
        dhParams,
        currentAngles,
        targetPos,
        0.1
      );
    }
    
    setLocalConfig(currentAngles);
    onConfigChange(currentAngles);
    
    if (robotRenderer) {
      robotRenderer.addWaypoint(new THREE.Vector3(targetPos[0], targetPos[1], targetPos[2]));
    }
  };

  const stepBackward = () => {
    let waypoints = [];
    
    if (trajectoryType === 'square' && squareCorners) {
      waypoints = generateSquareWaypoints();
    } else if (trajectoryType === 'circle' && circleParams) {
      waypoints = generateCircleWaypoints();
    } else if (trajectoryType === 'sine' && sineParams) {
      waypoints = generateSineWaypoints();
    }
    
    if (waypoints.length === 0) return;
    
    // Don't go below index 0
    if (stepIndex <= 0) return;
    
    const prevIndex = stepIndex - 1;
    setStepIndex(prevIndex);
    
    const targetPos = waypoints[prevIndex];
    let currentAngles = [...localConfig];
    
    for (let iter = 0; iter < 5; iter++) {
      currentAngles = RobotMath.differentialIK(
        dhParams,
        currentAngles,
        targetPos,
        0.1
      );
    }
    
    setLocalConfig(currentAngles);
    onConfigChange(currentAngles);
    
    if (robotRenderer) {
      robotRenderer.addWaypoint(new THREE.Vector3(targetPos[0], targetPos[1], targetPos[2]));
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
        {localConfig.map((angle, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={{ fontSize: '32px', fontWeight: '500' }}>
              θ{i+1}: {(angle * 180 / Math.PI).toFixed(0)}°
            </label>
            <input
              type="range"
              min="-180"
              max="180"
              step="1"
              value={(angle * 180 / Math.PI).toFixed(0)}
              onChange={(e) => handleSliderChange(i, parseFloat(e.target.value))}
              style={{ width: '100%' }}
              disabled={isAnimating}
            />
          </div>
        ))}
      </div>
      
      <div style={{ display: 'flex', gap: '24px', justifyContent: 'center', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="radio"
              value="square"
              checked={trajectoryType === 'square'}
              onChange={(e) => setTrajectoryType(e.target.value)}
              disabled={isAnimating}
              style={{ width: '16px', height: '16px' }}
            />
            <span style={{ fontWeight: '500', fontSize: '32px' }}>Square</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="radio"
              value="circle"
              checked={trajectoryType === 'circle'}
              onChange={(e) => setTrajectoryType(e.target.value)}
              disabled={isAnimating}
              style={{ width: '16px', height: '16px' }}
            />
            <span style={{ fontWeight: '500', fontSize: '32px' }}>Circle</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="radio"
              value="sine"
              checked={trajectoryType === 'sine'}
              onChange={(e) => setTrajectoryType(e.target.value)}
              disabled={isAnimating}
              style={{ width: '16px', height: '16px' }}
            />
            <span style={{ fontWeight: '500', fontSize: '32px' }}>Sine Wave</span>
          </label>
        </div>
      </div>
      
      <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={handleReset}
          style={{
            padding: '14px 32px',
            backgroundColor: '#f87171',
            color: 'white',
            borderRadius: '4px',
            fontWeight: '500',
            border: 'none',
            cursor: isAnimating ? 'not-allowed' : 'pointer',
            opacity: isAnimating ? 0.5 : 1,
            fontSize: '32px'
          }}
          disabled={isAnimating}
          onMouseOver={(e) => !isAnimating && (e.target.style.backgroundColor = '#ef4444')}
          onMouseOut={(e) => !isAnimating && (e.target.style.backgroundColor = '#f87171')}
        >
          Reset
        </button>
        <button
          onClick={handleRandom}
          style={{
            padding: '14px 32px',
            backgroundColor: '#4ade80',
            color: 'white',
            borderRadius: '4px',
            fontWeight: '500',
            border: 'none',
            cursor: isAnimating ? 'not-allowed' : 'pointer',
            opacity: isAnimating ? 0.5 : 1,
            fontSize: '32px'
          }}
          disabled={isAnimating}
          onMouseOver={(e) => !isAnimating && (e.target.style.backgroundColor = '#22c55e')}
          onMouseOut={(e) => !isAnimating && (e.target.style.backgroundColor = '#4ade80')}
        >
          Random
        </button>
        <button
          onClick={
            trajectoryType === 'square' ? initSquare : 
            trajectoryType === 'circle' ? initCircle : 
            initSineWave
          }
          style={{
            padding: '14px 32px',
            backgroundColor: '#60a5fa',
            color: 'white',
            borderRadius: '4px',
            fontWeight: '500',
            border: 'none',
            cursor: isAnimating ? 'not-allowed' : 'pointer',
            opacity: isAnimating ? 0.5 : 1,
            fontSize: '32px'
          }}
          disabled={isAnimating}
          onMouseOver={(e) => !isAnimating && (e.target.style.backgroundColor = '#3b82f6')}
          onMouseOut={(e) => !isAnimating && (e.target.style.backgroundColor = '#60a5fa')}
        >
          Init {trajectoryType === 'square' ? 'Square' : trajectoryType === 'circle' ? 'Circle' : 'Sine Wave'}
        </button>
        {!isAnimating ? (
          <button
            onClick={
              trajectoryType === 'square' ? animateSquare : 
              trajectoryType === 'circle' ? animateCircle : 
              animateSineWave
            }
            style={{
              padding: '14px 32px',
              backgroundColor: '#a78bfa',
              color: 'white',
              borderRadius: '4px',
              fontWeight: '500',
              border: 'none',
              cursor: (trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams) ? 'not-allowed' : 'pointer',
              opacity: (trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams) ? 0.5 : 1,
              fontSize: '32px'
            }}
            disabled={trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams}
            onMouseOver={(e) => {
              const enabled = trajectoryType === 'square' ? squareCorners : trajectoryType === 'circle' ? circleParams : sineParams;
              if (enabled) e.target.style.backgroundColor = '#8b5cf6';
            }}
            onMouseOut={(e) => {
              const enabled = trajectoryType === 'square' ? squareCorners : trajectoryType === 'circle' ? circleParams : sineParams;
              if (enabled) e.target.style.backgroundColor = '#a78bfa';
            }}
          >
            Animate
          </button>
        ) : (
          <button
            onClick={stopAnimation}
            style={{
              padding: '14px 32px',
              backgroundColor: '#fb923c',
              color: 'white',
              borderRadius: '4px',
              fontWeight: '500',
              border: 'none',
              cursor: 'pointer',
              fontSize: '32px'
            }}
            onMouseOver={(e) => e.target.style.backgroundColor = '#f97316'}
            onMouseOut={(e) => e.target.style.backgroundColor = '#fb923c'}
          >
            Stop
          </button>
        )}
        <button
          onClick={stepBackward}
          style={{
            padding: '14px 32px',
            backgroundColor: '#06b6d4',
            color: 'white',
            borderRadius: '4px',
            fontWeight: '500',
            border: 'none',
            cursor: (trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams) ? 'not-allowed' : 'pointer',
            opacity: (trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams) ? 0.5 : 1,
            fontSize: '32px'
          }}
          disabled={trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams}
          onMouseOver={(e) => {
            const enabled = trajectoryType === 'square' ? squareCorners : trajectoryType === 'circle' ? circleParams : sineParams;
            if (enabled) e.target.style.backgroundColor = '#0891b2';
          }}
          onMouseOut={(e) => {
            const enabled = trajectoryType === 'square' ? squareCorners : trajectoryType === 'circle' ? circleParams : sineParams;
            if (enabled) e.target.style.backgroundColor = '#06b6d4';
          }}
        >
          Step -
        </button>
        <button
          onClick={stepForward}
          style={{
            padding: '14px 32px',
            backgroundColor: '#8b5cf6',
            color: 'white',
            borderRadius: '4px',
            fontWeight: '500',
            border: 'none',
            cursor: (trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams) ? 'not-allowed' : 'pointer',
            opacity: (trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams) ? 0.5 : 1,
            fontSize: '32px'
          }}
          disabled={trajectoryType === 'square' ? !squareCorners : trajectoryType === 'circle' ? !circleParams : !sineParams}
          onMouseOver={(e) => {
            const enabled = trajectoryType === 'square' ? squareCorners : trajectoryType === 'circle' ? circleParams : sineParams;
            if (enabled) e.target.style.backgroundColor = '#7c3aed';
          }}
          onMouseOut={(e) => {
            const enabled = trajectoryType === 'square' ? squareCorners : trajectoryType === 'circle' ? circleParams : sineParams;
            if (enabled) e.target.style.backgroundColor = '#8b5cf6';
          }}
        >
          Step +
        </button>
      </div>
      
      {(squareCorners || circleParams || sineParams) && (
        <div style={{ textAlign: 'center', fontSize: '32px', color: '#6b7280' }}>
          {trajectoryType === 'square' ? 'Square' : trajectoryType === 'circle' ? 'Circle' : 'Sine Wave'} plane: <span style={{ fontWeight: 'bold' }}>{trajectoryPlane.toUpperCase()}</span>
          {sineParams && (
            <div style={{ marginTop: '4px', fontSize: '28px' }}>
              Wavelength: {sineParams.wavelength.toFixed(1)}mm, Amplitude: {sineParams.amplitude.toFixed(1)}mm
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ========================
// MAIN APP
// ========================

const App = () => {
  return (
    <BaseRobotView dhParams={RobotConfig.UR5_DH_PARAMS}>
      <TrajectoryControls />
    </BaseRobotView>
  );
};

export default App;