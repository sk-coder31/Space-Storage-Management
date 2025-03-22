# Space-Storage-Management
# API Documentation

Welcome to the API documentation! This README provides details on the available endpoints, their functionalities, and how to use them.

---

## Endpoints Overview

### **1. Placement Recommendations**
**POST /api/placement**  
- **Description:** Main endpoint for placement recommendations.  
- **Input:** List of items and containers.  
- **Output:** Returns placements and potential rearrangements.

---

### **2. Create Container**
**POST /api/containers/create**  
- **Description:** Creates a new container.  
- **Input:** Container details.  
- **Output:** Returns success status and container ID.

---

### **3. Deactivate Container**
**POST /api/containers/deactivate/{container_id}**  
- **Description:** Deactivates an existing container.  
- **Input:** Path parameter for `container_id`. No request body required.  
- **Output:** Returns success status.

---

### **4. Reactivate Container**
**POST /api/containers/reactivate/{container_id}**  
- **Description:** Reactivates a previously deactivated container.  
- **Input:** Path parameter for `container_id`. No request body required.  
- **Output:** Returns success status.

---

### **5. Container Statistics**
**GET /api/containers/stats**  
- **Description:** Retrieves statistics for containers.  
- **Input:** Optional query parameter for specific `container_id`. No request body required.  
- **Output:** Returns container statistics.

---

### **6. Generate Container ID**
**POST /api/containers/generate-id**  
- **Description:** Generates a new container ID.  
- **Input:** No request body required.  
- **Output:** Returns a newly generated container ID.

---

### **7. Get Item Placement Details**
**GET /api/items/{item_id}**  
- **Description:** Retrieves placement details for an item.  
- **Input:** Path parameter for `item_id`. No request body required.  
- **Output:** Returns placement details.

---

### **8. Delete Item**
**DELETE /api/items/{item_id}**  
- **Description:** Deletes an item.  
- **Input:** Path parameter for `item_id`. No request body required.  
- **Output:** Returns success status.

---

## Usage Notes

- Ensure that you use the correct HTTP method (GET, POST, DELETE) for the corresponding endpoint.
- All endpoints expect proper input parameters to function as described.
- For any issues or questions, please contact the API support team.

---

This README is structured to give clear guidance on how to use the API effectively. Let me know if you'd like further customization or additional information included!
