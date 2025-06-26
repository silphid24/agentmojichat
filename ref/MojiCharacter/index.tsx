import React from 'react';

export interface MojiCharacterProps {
  size?: number;
  className?: string;
  faceColor?: string;
  eyeColor?: string;
  mouthColor?: string;
  cheekColor?: string;
  cheekOpacity?: number;
}

const MojiCharacter: React.FC<MojiCharacterProps> = ({
  size = 32,
  className = '',
  faceColor = '#f8f9fa',
  eyeColor = '#1f2937',
  mouthColor = '#1f2937',
  cheekColor = '#fca5a5',
  cheekOpacity = 0.6,
}) => {
  const strokeWidth = size <= 32 ? 1.2 : 1.5;
  const eyeRadius = size * 0.06;
  const cheekRadius = size * 0.08;
  const mouthStrokeWidth = size * 0.05;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`drop-shadow-sm ${className}`}
    >
      {/* 모지 캐릭터 얼굴 */}
      <circle cx={size / 2} cy={size / 2} r={size * 0.45} fill={faceColor} stroke="#9ca3af" strokeWidth={strokeWidth} />

      {/* 왼쪽 눈 */}
      <circle cx={size * 0.35} cy={size * 0.4} r={eyeRadius} fill={eyeColor} />

      {/* 오른쪽 눈 */}
      <circle cx={size * 0.65} cy={size * 0.4} r={eyeRadius} fill={eyeColor} />

      {/* 입 (미소) */}
      <path
        d={`M${size * 0.35} ${size * 0.6} Q${size * 0.5} ${size * 0.75} ${size * 0.65} ${size * 0.6}`}
        stroke={mouthColor}
        strokeWidth={mouthStrokeWidth}
        fill="none"
        strokeLinecap="round"
      />

      {/* 왼쪽 볼 */}
      <circle cx={size * 0.25} cy={size * 0.55} r={cheekRadius} fill={cheekColor} opacity={cheekOpacity} />

      {/* 오른쪽 볼 */}
      <circle cx={size * 0.75} cy={size * 0.55} r={cheekRadius} fill={cheekColor} opacity={cheekOpacity} />
    </svg>
  );
};

export default MojiCharacter;
