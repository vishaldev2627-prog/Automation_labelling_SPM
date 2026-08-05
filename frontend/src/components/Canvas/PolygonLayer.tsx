import Konva from "konva";
import { Circle, Group, Line } from "react-konva";
import { useAnnotationStore } from "../../store/annotationStore";
import type { AnnotationObject, Point } from "../../types";

interface PolygonLayerProps {
  obj: AnnotationObject;
  color: string;
  imageWidth: number;
  imageHeight: number;
  isSelected: boolean;
  editable: boolean;
  onSelect: () => void;
}

/** Renders one object's polygon outline, optionally with draggable/editable vertices. */
export default function PolygonLayer({
  obj,
  color,
  imageWidth,
  imageHeight,
  isSelected,
  editable,
  onSelect,
}: PolygonLayerProps) {
  const updateObject = useAnnotationStore((s) => s.updateObject);

  if (!obj.visible || obj.polygon.length < 3) return null;

  const pixelPoints = obj.polygon.flatMap((p) => [p.x * imageWidth, p.y * imageHeight]);

  const commitPolygon = (newPoints: Point[]) => {
    updateObject(obj.id, (o) => ({ ...o, polygon: newPoints, status: "edited" }));
  };

  const handleVertexDragEnd = (index: number, e: Konva.KonvaEventObject<DragEvent>) => {
    const node = e.target;
    const nx = node.x() / imageWidth;
    const ny = node.y() / imageHeight;
    const clamped: Point = { x: Math.min(1, Math.max(0, nx)), y: Math.min(1, Math.max(0, ny)) };
    const newPoints = obj.polygon.map((p, i) => (i === index ? clamped : p));
    commitPolygon(newPoints);
  };

  const handleVertexContextMenu = (index: number, e: Konva.KonvaEventObject<PointerEvent>) => {
    e.evt.preventDefault();
    if (obj.polygon.length <= 3) return; // keep polygon valid
    const newPoints = obj.polygon.filter((_, i) => i !== index);
    commitPolygon(newPoints);
  };

  const handleLineDblClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!editable) return;
    const stage = e.target.getStage();
    if (!stage) return;
    const pos = stage.getRelativePointerPosition();
    if (!pos) return;

    // find nearest segment to insert a vertex on
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < obj.polygon.length; i++) {
      const a = obj.polygon[i];
      const b = obj.polygon[(i + 1) % obj.polygon.length];
      const ax = a.x * imageWidth,
        ay = a.y * imageHeight;
      const bx = b.x * imageWidth,
        by = b.y * imageHeight;
      const t = Math.max(0, Math.min(1, ((pos.x - ax) * (bx - ax) + (pos.y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 || 1)));
      const px = ax + t * (bx - ax);
      const py = ay + t * (by - ay);
      const dist = (pos.x - px) ** 2 + (pos.y - py) ** 2;
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    const newPoint: Point = { x: pos.x / imageWidth, y: pos.y / imageHeight };
    const newPoints = [...obj.polygon.slice(0, bestIdx + 1), newPoint, ...obj.polygon.slice(bestIdx + 1)];
    commitPolygon(newPoints);
  };

  const handleGroupDragEnd = (e: Konva.KonvaEventObject<DragEvent>) => {
    const dx = e.target.x() / imageWidth;
    const dy = e.target.y() / imageHeight;
    if (dx === 0 && dy === 0) return;
    const newPoints = obj.polygon.map((p) => ({
      x: Math.min(1, Math.max(0, p.x + dx)),
      y: Math.min(1, Math.max(0, p.y + dy)),
    }));
    e.target.position({ x: 0, y: 0 });
    commitPolygon(newPoints);
  };

  return (
    <Group>
      {/* Extra disjoint pieces of the same object — only ever present for
          fine_structure classes, where a branching or segmented crack is one
          object made of several contours. Drawn so an annotator can actually
          see and check them; not vertex-editable, since editing which piece is
          "the" polygon has no meaning. Dashed to distinguish them from the
          primary piece, which is the one editing acts on. */}
      {obj.extra_polygons.map((piece, i) =>
        piece.length >= 3 ? (
          <Line
            key={`extra-${i}`}
            points={piece.flatMap((p) => [p.x * imageWidth, p.y * imageHeight])}
            closed
            stroke={color}
            strokeWidth={isSelected ? 2 : 1.25}
            dash={[4, 3]}
            fillEnabled={false}
            listening={false}
          />
        ) : null,
      )}
      <Line
        points={pixelPoints}
        closed
        stroke={color}
        strokeWidth={isSelected ? 2.5 : 1.5}
        fill={color}
        opacity={1}
        fillEnabled={false}
        onClick={onSelect}
        onTap={onSelect}
        onDblClick={handleLineDblClick}
        draggable={editable && isSelected}
        onDragEnd={handleGroupDragEnd}
        hitStrokeWidth={12}
      />
      {editable &&
        isSelected &&
        obj.polygon.map((p, i) => (
          <Circle
            key={i}
            x={p.x * imageWidth}
            y={p.y * imageHeight}
            radius={5}
            fill="#ffffff"
            stroke={color}
            strokeWidth={2}
            draggable
            onDragMove={(e) => e.cancelBubble = true}
            onDragEnd={(e) => handleVertexDragEnd(i, e)}
            onContextMenu={(e) => handleVertexContextMenu(i, e)}
          />
        ))}
    </Group>
  );
}
