import Konva from "konva";
import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";
import toast from "react-hot-toast";
import useImage from "use-image";
import { ImagesAPI } from "../../api/client";
import { useAnnotationStore } from "../../store/annotationStore";
import { useDatasetStore } from "../../store/datasetStore";
import { useSettingsStore } from "../../store/settingsStore";
import PolygonLayer from "./PolygonLayer";

const MISSING_CLASS_COLOR = "#9ca3af";
const MIN_DRAW_BOX_PX = 6;

export default function AnnotationCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);

  const { imageId, imageWidth, imageHeight, objects, selectedObjectId, selectObject, addPendingPoint, createObjectFromBox } =
    useAnnotationStore();
  const pendingPositive = useAnnotationStore((s) => s.pendingPositivePoints);
  const pendingNegative = useAnnotationStore((s) => s.pendingNegativePoints);
  const classes = useDatasetStore((s) => s.classes);
  const {
    showBoundingBox,
    showMask,
    showPolygon,
    showImage,
    maskOpacity,
    toolMode,
    hiddenClassIds,
    activeClassId,
    zoom,
    panX,
    panY,
    setZoom,
    setPan,
    setToolMode,
  } = useSettingsStore();

  const imageUrl = imageId ? ImagesAPI.fileUrl(imageId) : "";
  const [image] = useImage(imageUrl, "anonymous");

  const colorForClass = useMemo(() => {
    const map = new Map<number, string>();
    classes.forEach((c) => map.set(c.class_id, c.color));
    return map;
  }, [classes]);

  const visibleObjects = useMemo(
    () => objects.filter((o) => o.visible && o.status !== "rejected" && !hiddenClassIds.has(o.class_id)),
    [objects, hiddenClassIds],
  );

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setContainerSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // fit-to-screen whenever a new image loads
  useEffect(() => {
    if (imageWidth && imageHeight && containerSize.width && containerSize.height) {
      const scale = Math.min(containerSize.width / imageWidth, containerSize.height / imageHeight) * 0.95;
      setZoom(scale);
      setPan((containerSize.width - imageWidth * scale) / 2, (containerSize.height - imageHeight * scale) / 2);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageId, imageWidth, imageHeight]);

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const oldScale = zoom;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const mousePointTo = {
      x: (pointer.x - panX) / oldScale,
      y: (pointer.y - panY) / oldScale,
    };

    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const scaleBy = 1.08;
    const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;
    const clamped = Math.min(8, Math.max(0.05, newScale));

    setZoom(clamped);
    setPan(pointer.x - mousePointTo.x * clamped, pointer.y - mousePointTo.y * clamped);
  };

  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;

    if (toolMode === "positive-click" || toolMode === "negative-click") {
      const pos = stage.getRelativePointerPosition();
      if (!pos || !imageWidth || !imageHeight) return;
      addPendingPoint({ x: pos.x / imageWidth, y: pos.y / imageHeight }, toolMode === "positive-click");
      return;
    }

    if (e.target === stage) {
      selectObject(null);
    }
  };

  const handleDrawMouseDown = () => {
    if (toolMode !== "draw-box") return;
    const stage = stageRef.current;
    const pos = stage?.getRelativePointerPosition();
    if (!pos) return;
    setDrawStart(pos);
    setDrawCurrent(pos);
  };

  const handleDrawMouseMove = () => {
    if (toolMode !== "draw-box" || !drawStart) return;
    const stage = stageRef.current;
    const pos = stage?.getRelativePointerPosition();
    if (!pos) return;
    setDrawCurrent(pos);
  };

  const handleDrawMouseUp = async () => {
    if (toolMode !== "draw-box" || !drawStart || !drawCurrent) {
      setDrawStart(null);
      setDrawCurrent(null);
      return;
    }
    const x1 = Math.min(drawStart.x, drawCurrent.x);
    const y1 = Math.min(drawStart.y, drawCurrent.y);
    const wPx = Math.abs(drawCurrent.x - drawStart.x);
    const hPx = Math.abs(drawCurrent.y - drawStart.y);
    setDrawStart(null);
    setDrawCurrent(null);

    if (wPx < MIN_DRAW_BOX_PX || hPx < MIN_DRAW_BOX_PX || !imageWidth || !imageHeight) return;
    if (activeClassId == null) {
      toast.error("Pick a class in the sidebar first, then draw the box");
      return;
    }
    const className = classes.find((c) => c.class_id === activeClassId)?.name ?? "";
    const bbox = {
      x_center: (x1 + wPx / 2) / imageWidth,
      y_center: (y1 + hPx / 2) / imageHeight,
      width: wPx / imageWidth,
      height: hPx / imageHeight,
    };

    try {
      await toast.promise(createObjectFromBox(bbox, activeClassId, className), {
        loading: "Generating mask...",
        success: `Added ${className}`,
        error: "Mask generation failed",
      });
      setToolMode("select");
    } catch {
      /* surfaced via toast */
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full overflow-hidden bg-surface-950"
      style={{ cursor: toolMode === "draw-box" ? "crosshair" : undefined }}
    >
      <Stage
        ref={stageRef}
        width={containerSize.width}
        height={containerSize.height}
        scaleX={zoom}
        scaleY={zoom}
        x={panX}
        y={panY}
        draggable={toolMode === "pan"}
        onWheel={handleWheel}
        onClick={handleStageClick}
        onMouseDown={handleDrawMouseDown}
        onMouseMove={handleDrawMouseMove}
        onMouseUp={handleDrawMouseUp}
        onDragEnd={(e) => {
          if (toolMode === "pan") setPan(e.target.x(), e.target.y());
        }}
        className="konva-stage-container"
      >
        <Layer listening={false}>{showImage && image && <KonvaImage image={image} />}</Layer>

        {/* semi-transparent mask fill, approximated from the current polygon */}
        <Layer listening={false} opacity={maskOpacity}>
          {showMask &&
            visibleObjects.map((o) => {
              if (o.polygon.length < 3) return null;
              const color = colorForClass.get(o.class_id) ?? MISSING_CLASS_COLOR;
              const points = o.polygon.flatMap((p) => [p.x * imageWidth, p.y * imageHeight]);
              return <Line key={`fill-${o.id}`} points={points} closed fill={color} />;
            })}
        </Layer>

        <Layer listening={false}>
          {showBoundingBox &&
            visibleObjects.map((o) => {
              const color = colorForClass.get(o.class_id) ?? MISSING_CLASS_COLOR;
              const x1 = (o.bbox.x_center - o.bbox.width / 2) * imageWidth;
              const y1 = (o.bbox.y_center - o.bbox.height / 2) * imageHeight;
              return (
                <Rect
                  key={`bbox-${o.id}`}
                  x={x1}
                  y={y1}
                  width={o.bbox.width * imageWidth}
                  height={o.bbox.height * imageHeight}
                  stroke={color}
                  strokeWidth={1}
                  dash={[6, 4]}
                  opacity={0.85}
                />
              );
            })}
        </Layer>

        <Layer>
          {showPolygon &&
            visibleObjects.map((o) => (
              <PolygonLayer
                key={o.id}
                obj={o}
                color={colorForClass.get(o.class_id) ?? MISSING_CLASS_COLOR}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                isSelected={selectedObjectId === o.id}
                editable={toolMode === "select" || toolMode === "edit-vertex"}
                onSelect={() => selectObject(o.id)}
              />
            ))}
        </Layer>

        <Layer listening={false}>
          {pendingPositive.map((p, i) => (
            <Circle key={`pp-${i}`} x={p.x * imageWidth} y={p.y * imageHeight} radius={5} fill="#22c55e" stroke="#fff" strokeWidth={1} />
          ))}
          {pendingNegative.map((p, i) => (
            <Circle key={`np-${i}`} x={p.x * imageWidth} y={p.y * imageHeight} radius={5} fill="#ef4444" stroke="#fff" strokeWidth={1} />
          ))}
          {toolMode === "draw-box" && drawStart && drawCurrent && (
            <Rect
              x={Math.min(drawStart.x, drawCurrent.x)}
              y={Math.min(drawStart.y, drawCurrent.y)}
              width={Math.abs(drawCurrent.x - drawStart.x)}
              height={Math.abs(drawCurrent.y - drawStart.y)}
              stroke={(activeClassId != null && colorForClass.get(activeClassId)) || "#3b82f6"}
              strokeWidth={1.5}
              dash={[6, 4]}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
