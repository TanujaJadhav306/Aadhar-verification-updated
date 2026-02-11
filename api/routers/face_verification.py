"""
Face Verification API endpoints
Face matching service using OpenCV YuNet + SFace (ArcFace)
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.opencv_sface import (
    basic_liveness,
    decode_image,
    cosine_similarity_sface,
    detect_faces,
    detect_best_face_for_document,
    best_match_same_image_selfie_with_card,
    face_feature,
    pick_largest_face_index,
)
import os
import cv2
import numpy as np

# Ensure uploads directory exists - Pointing to Backend Service Uploads
# Path: ../../../AmpTalentIQ-backend/amptalentiq-api/uploads/verification
# We use absolute path to ensure it works regardless of CWD
BACKEND_UPLOADS_DIR = r"c:\Users\tanuja.jadhav\OneDrive - Ampcus Tech Pvt Ltd\Desktop\AmpTalentIQ\AmpTalentIQ-backend\amptalentiq-api\uploads\verification"
UPLOAD_DIR = BACKEND_UPLOADS_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()


@router.post("/face-verification/liveness")
async def liveness(file: UploadFile = File(...)) -> dict:
    """
    Basic liveness detection endpoint.
    Checks if image contains exactly one face with reasonable quality.
    """
    try:
        img = decode_image(await file.read())
        return basic_liveness(img)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/face-verification/detect")
async def detect(
    file: UploadFile = File(...), 
    score_threshold: float = 0.6,
    candidateExamId: str = Form(...),
    imageType: str = Form("candidate"),
    candidateName: str | None = Form(default=None)
) -> dict:
    """
    Face detection only (for UI feedback).
    Returns detected face boxes and scores.
    """
    try:
        contents = await file.read()
        img = decode_image(contents)
        
        # Save image
        if candidateExamId:
            # Create subfolder for candidate
            candidate_dir = os.path.join(UPLOAD_DIR, str(candidateExamId))
            os.makedirs(candidate_dir, exist_ok=True)
            
            # Generate filename with timestamp and candidate name
            import time
            timestamp = int(time.time())
            
            # Sanitize candidate name
            safe_name = "candidate"
            if candidateName:
                # Remove special characters and spaces
                safe_name = "".join(c for c in candidateName if c.isalnum() or c in ('-', '_'))
            
            filename = f"{imageType}-{safe_name}-{timestamp}.jpg"
            filepath = os.path.join(candidate_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(contents)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    faces_raw, infos = detect_faces(img, score_threshold=score_threshold)
    boxes = []
    for i, info in enumerate(infos):
        x, y, w, h = info.box
        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "score": float(info.score),
            }
        )

    return {
        "ok": True, 
        "faceCount": len(infos), 
        "boxes": boxes,
        "candidateExamId": candidateExamId
    }


@router.post("/face-verification/verify")
async def verify(
    document: UploadFile = File(...),
    selfie: UploadFile = File(...),
    candidateExamId: str = Form(...),
    threshold: float = 0.363,
    run_liveness: bool = True,
    candidateName: str | None = Form(default=None),
) -> dict:
    """
    Face verification endpoint.
    Compares a face from a document (Aadhaar/PAN/DL) with a selfie.
    """
    try:
        doc_content = await document.read()
        selfie_content = await selfie.read()
        
        doc_img = decode_image(doc_content)
        selfie_img = decode_image(selfie_content)
        
        # Save images
        if candidateExamId:
            # Create subfolder for candidate
            candidate_dir = os.path.join(UPLOAD_DIR, str(candidateExamId))
            os.makedirs(candidate_dir, exist_ok=True)

            # Generate filename with timestamp and candidate name
            import time
            timestamp = int(time.time())
            
            # Sanitize candidate name
            safe_name = "candidate"
            if candidateName:
                # Remove special characters and spaces
                safe_name = "".join(c for c in candidateName if c.isalnum() or c in ('-', '_'))

            # Save Document Image
            doc_filename = f"document-{safe_name}-{timestamp}.jpg"
            doc_filepath = os.path.join(candidate_dir, doc_filename)
            with open(doc_filepath, "wb") as f:
                f.write(doc_content)
                
            # Save Candidate Image (Selfie)
            selfie_filename = f"candidate-{safe_name}-{timestamp}.jpg"
            selfie_filepath = os.path.join(candidate_dir, selfie_filename)
            with open(selfie_filepath, "wb") as f:
                f.write(selfie_content)
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    doc_raw, doc_infos = detect_faces(doc_img)
    selfie_raw, selfie_infos = detect_faces(selfie_img)
    doc_i = pick_largest_face_index(doc_infos)
    selfie_i = pick_largest_face_index(selfie_infos)

    doc_used_meta = None
    doc_used_img = doc_img
    doc_used_infos = doc_infos

    if doc_i is None:
        found = detect_best_face_for_document(doc_img)
        if found is None:
            raise HTTPException(status_code=422, detail="No face detected in document image.")
        doc_used_img, doc_face_row, doc_used_infos, doc_used_meta = found
    else:
        doc_face_row = doc_raw[doc_i]

    if selfie_i is None:
        raise HTTPException(status_code=422, detail="No face detected in selfie image.")

    doc_feat = face_feature(doc_used_img, doc_face_row)
    selfie_feat = face_feature(selfie_img, selfie_raw[selfie_i])
    sim = cosine_similarity_sface(doc_feat, selfie_feat)
    is_match = sim >= threshold

    # UI %: cosine similarity is in [-1..1], clamp to [0..1]
    match_percent = int(round(max(0.0, min(1.0, (sim + 1.0) / 2.0)) * 100))

    liveness_result = basic_liveness(selfie_img) if run_liveness else None

    # Connect and update database
    update_candidate_verification_in_db(candidateExamId, is_match, match_percent)

    return {
        "ok": True,
        "engine": "OpenCV YuNet + SFace (opencv_zoo ONNX)",
        "similarity": float(sim),
        "threshold": float(threshold),
        "isMatch": bool(is_match),
        "matchPercent": match_percent,
        "candidateExamId": candidateExamId,
        "faces": {
            "documentFaceCount": len(doc_used_infos),
            "selfieFaceCount": len(selfie_infos),
        },
        "documentSearch": doc_used_meta,
        "liveness": liveness_result,
        "note": "For production, use a real liveness/anti-spoof model + audited thresholds.",
    }


def update_candidate_verification_in_db(candidate_exam_id, is_match, match_percent):
    """Update candidate verification status in PostgreSQL"""
    try:
        from services.database_service import DatabaseService
        query = """
            UPDATE "exams-candidate"
            SET "isVerification" = %s, "matchPercent" = %s
            WHERE id = %s
        """
        affected = DatabaseService.execute_query(query, (is_match, match_percent, candidate_exam_id))
        print(f"Database updated for candidate {candidate_exam_id}: {affected} rows affected.")
        return True
    except Exception as e:
        print(f"Failed to update database: {e}")
        return False


@router.post("/face-verification/verify-single")
async def verify_single(
    file: UploadFile = File(...),
    threshold: float = 0.363,
    candidate_box: str | None = Form(default=None),
    document_box: str | None = Form(default=None),
) -> dict:
    """
    Single image mode: user holds Aadhaar/PAN/DL card in the same photo.
    We detect multiple faces and match the largest face (candidate) with a smaller face (ID portrait).
    """
    try:
        img = decode_image(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    result = best_match_same_image_selfie_with_card(
        img,
        threshold=threshold,
        candidate_box_json=candidate_box,
        document_box_json=document_box,
    )
    if not result.get("ok", False):
        raise HTTPException(status_code=422, detail=result.get("error", "Verify failed."))
    return result
