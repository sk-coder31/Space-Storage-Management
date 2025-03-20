import csv
import io
import random
import json
import time
from datetime import datetime, timedelta
from http.client import HTTPException
from typing import List, Dict, Tuple, Optional, Set, Any
from urllib import request

from scipy.spatial import cKDTree
import concurrent.futures
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn
import numpy as np
from collections import defaultdict

from starlette.responses import FileResponse

app = FastAPI(title="ISS Cargo Management System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global storage for our simulation
class GlobalState:
    def __init__(self):
        self.containers = {}  # Container ID -> Container object
        self.items = {}  # Item ID -> Item object
        self.logs = []
        self.current_date = datetime.now()
        self.kdtree = None
        self.kdtree_data = []


global_state = GlobalState()


# Optimized data models using numpy arrays for space efficiency
class Coordinates(BaseModel):
    width: float
    depth: float
    height: float


class Position(BaseModel):
    startCoordinates: Coordinates
    endCoordinates: Coordinates


class ItemToUse:
    pass


class Item(BaseModel):
    itemId: str
    name: str
    width: float
    depth: float
    height: float
    priority: int
    expiryDate: Optional[str] = None
    usageLimit: int
    preferredZone: str

    class Container(BaseModel):
        containerId: str
        zone: str
        width: float
        depth: float
        height: float

    class ItemToUse(BaseModel):
        itemId: Optional[str] = None
        name: Optional[str] = None

    class TimeSimulationRequest(BaseModel):
        numOfDays: Optional[int] = None
        toTimestamp: Optional[str] = None
        itemsToBeUsedPerDay: List[ItemToUse]

        class LogQuery(BaseModel):
            startDate: str
            endDate: str
            itemId: Optional[str] = None
            userId: Optional[str] = None
            actionType: Optional[str] = None


def check_item_status(item_id: str, current_date_str: str):
    """Check if an item is expired or depleted"""
    if item_id not in items_db:
        return None, None

    item = items_db[item_id]
    is_expired = False
    is_depleted = False

    # Check expiry
    if item.get("expiryDate") and item["expiryDate"] != "N/A":
        expiry_date = datetime.fromisoformat(item["expiryDate"].replace("Z", "+00:00"))
        current_date = datetime.fromisoformat(current_date_str.replace("Z", "+00:00"))
        if current_date >= expiry_date:
            is_expired = True

    # Check usage limit
    if item["remainingUses"] <= 0:
        is_depleted = True

    return is_expired, is_depleted


# Time Simulation API
@app.post("/api/simulate/day")
async def simulate_day(request: TimeSimulationRequest):
    global current_date

    current_datetime = datetime.fromisoformat(current_date.replace("Z", "+00:00"))

    # Determine the target date
    if request.numOfDays:
        target_datetime = current_datetime + timedelta(days=request.numOfDays)
    elif request.toTimestamp:
        target_datetime = datetime.fromisoformat(request.toTimestamp.replace("Z", "+00:00"))
    else:
        # Default to 1 day if neither is specified
        target_datetime = current_datetime + timedelta(days=1)

    days_difference = (target_datetime - current_datetime).days

    items_used = []
    items_expired = []
    items_depleted = []

    # Process each day
    for _ in range(days_difference):
        current_datetime += timedelta(days=1)

        # Process items to be used each day
        for item_to_use in request.itemsToBeUsedPerDay:
            # Find the item by ID or name
            found_item_id = None

            if item_to_use.itemId:
                if item_to_use.itemId in items_db:
                    found_item_id = item_to_use.itemId
            elif item_to_use.name:
                # Search by name
                for item_id, item in items_db.items():
                    if item["name"] == item_to_use.name:
                        found_item_id = item_id
                        break

            if found_item_id:
                # Use the item (decrease remaining uses)
                items_db[found_item_id]["remainingUses"] -= 1

                # Record usage
                items_used.append({
                    "itemId": found_item_id,
                    "name": items_db[found_item_id]["name"],
                    "remainingUses": items_db[found_item_id]["remainingUses"]
                })

                # Check if item is now depleted
                if items_db[found_item_id]["remainingUses"] <= 0:
                    items_depleted.append({
                        "itemId": found_item_id,
                        "name": items_db[found_item_id]["name"]
                    })

        # Check for expired items
        current_date_str = current_datetime.isoformat()
        for item_id, item in items_db.items():
            if item.get("expiryDate") and item["expiryDate"] != "N/A":
                expiry_date = datetime.fromisoformat(item["expiryDate"].replace("Z", "+00:00"))
                if current_datetime >= expiry_date and item_id not in [i["itemId"] for i in items_expired]:
                    items_expired.append({
                        "itemId": item_id,
                        "name": item["name"]
                    })

    # Update the current date
    current_date = current_datetime.isoformat()

    # Return the simulation results
    return {
        "success": True,
        "newDate": current_date,
        "changes": {
            "itemsUsed": items_used,
            "itemsExpired": items_expired,
            "itemsDepletedToday": items_depleted
        }
    }


def log_action(user_id: str, action_type: str, item_id: str, details: Dict[str, Any] = None):
    """Log an action in the system"""
    if details is None:
        details = {}

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "userId": user_id,
        "actionType": action_type,
        "itemId": item_id,
        "details": details
    }
    logs_db.append(log_entry)
    return log_entry
items_db = {}
containers_db = {}
item_placements = {}  # Maps itemId to {containerId, position}
logs_db = []
current_date = datetime.now().isoformat()

# Optimized Item class
class Item:
    def __init__(self, item_id: str, name: str, width: int, depth: int, height: int,
                 priority: int, expiry_date: Optional[str] = None, usage_limit: Optional[int] = None,
                 preferred_zone: Optional[str] = None, position: Optional[Tuple[int, int, int]] = None,
                 mass: float = 1.0, container_id: Optional[str] = None):
        self.item_id = item_id
        self.name = name
        self.width = width
        self.depth = depth
        self.height = height
        self.priority = priority
        self.expiry_date = datetime.fromisoformat(expiry_date) if expiry_date and expiry_date != "N/A" else None
        self.usage_limit = usage_limit
        self.preferred_zone = preferred_zone
        self.position = position or (0, 0, 0)
        self.mass = mass
        self.container_id = container_id
        self.remaining_uses = usage_limit

    def __lt__(self, other):
        return self.priority < other.priority

    def is_waste(self, current_date: datetime) -> Tuple[bool, str]:
        """Check if item is waste and return reason"""
        if self.expiry_date and current_date > self.expiry_date:
            return True, "Expired"
        if self.remaining_uses is not None and self.remaining_uses <= 0:
            return True, "Out of Uses"
        return False, ""

    def to_dict(self) -> Dict:
        """Convert item to dictionary representation"""
        return {
            "itemId": self.item_id,
            "name": self.name,
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "priority": self.priority,
            "expiryDate": self.expiry_date.isoformat() if self.expiry_date else None,
            "usageLimit": self.usage_limit,
            "remainingUses": self.remaining_uses,
            "preferredZone": self.preferred_zone,
            "mass": self.mass,
            "containerId": self.container_id,
            "position": self.position
        }


# Optimized Container class using sparse matrix representation
class Container:
    def __init__(self, container_id: str, width: int, depth: int, height: int, zone: str = "General"):
        self.container_id = container_id
        self.width = width
        self.depth = depth
        self.height = height
        self.zone = zone
        self.items = []

        # Using numpy sparse representation for efficiency
        self.space_matrix = np.zeros((width, depth, height), dtype=bool)

    def add_item(self, item: Item):
        """Add an item to the container and update space matrix"""
        self.items.append(item)
        item.container_id = self.container_id

        # Update the space matrix
        x, y, z = item.position
        self.space_matrix[x:x + item.width, y:y + item.depth, z:z + item.height] = True

    def remove_item(self, item: Item):
        """Remove an item from the container and update space matrix"""
        if item in self.items:
            self.items.remove(item)

            # Clear the space in matrix
            x, y, z = item.position
            self.space_matrix[x:x + item.width, y:y + item.depth, z:z + item.height] = False

            item.container_id = None
            return True
        return False

    def get_free_volume(self) -> int:
        """Get available free volume in the container"""
        used_volume = np.sum(self.space_matrix)
        total_volume = self.width * self.depth * self.height
        return total_volume - used_volume

    def can_fit_at_position(self, item: Item, position: Tuple[int, int, int]) -> bool:
        """Check if an item can fit at a specific position"""
        x, y, z = position

        # Check boundaries
        if x + item.width > self.width or y + item.depth > self.depth or z + item.height > self.height:
            return False

        # Check if space is free
        return not np.any(self.space_matrix[x:x + item.width, y:y + item.depth, z:z + item.height])

    def find_position(self, item: Item) -> Optional[Tuple[int, int, int]]:
        """Find a position where the item can fit using a more efficient algorithm"""
        # First try corners to maximize space usage
        for x in range(0, self.width - item.width + 1, max(1, item.width // 2)):
            for y in range(0, self.depth - item.depth + 1, max(1, item.depth // 2)):
                for z in range(0, self.height - item.height + 1, max(1, item.height // 2)):
                    if self.can_fit_at_position(item, (x, y, z)):
                        return (x, y, z)

        # If corners don't work, try a more exhaustive search but with stride
        stride_w = max(1, item.width // 4)
        stride_d = max(1, item.depth // 4)
        stride_h = max(1, item.height // 4)

        for x in range(0, self.width - item.width + 1, stride_w):
            for y in range(0, self.depth - item.depth + 1, stride_d):
                for z in range(0, self.height - item.height + 1, stride_h):
                    if self.can_fit_at_position(item, (x, y, z)):
                        return (x, y, z)

        # Last resort - exhaustive search
        for x in range(0, self.width - item.width + 1):
            for y in range(0, self.depth - item.depth + 1):
                for z in range(0, self.height - item.height + 1):
                    if self.can_fit_at_position(item, (x, y, z)):
                        return (x, y, z)

        return None

    def get_retrieval_steps(self, target_item: Item) -> List[Dict]:
        """Get steps needed to retrieve an item"""
        if target_item not in self.items:
            return []

        steps = []
        step_counter = 1

        # Get items that need to be moved to retrieve the target
        items_to_move = []
        tx, ty, tz = target_item.position
        target_width, target_depth, target_height = target_item.width, target_item.depth, target_item.height

        # Check if item is directly accessible (at depth 0)
        if ty == 0:
            steps.append({
                "step": step_counter,
                "action": "retrieve",
                "itemId": target_item.item_id,
                "itemName": target_item.name
            })
            return steps

        # Find items blocking this one
        for item in self.items:
            if item == target_item:
                continue

            ix, iy, iz = item.position
            item_width, item_depth, item_height = item.width, item.depth, item.height

            # Check if item is in front of our target
            if (iy < ty and
                    ix < tx + target_width and
                    ix + item_width > tx and
                    iz < tz + target_height and
                    iz + item_height > tz):
                items_to_move.append(item)

        # Process items that need to be moved
        for item in items_to_move:
            step_counter += 1
            steps.append({
                "step": step_counter,
                "action": "remove",
                "itemId": item.item_id,
                "itemName": item.name
            })

        # Now retrieve the target
        steps.append({
            "step": step_counter + 1,
            "action": "retrieve",
            "itemId": target_item.item_id,
            "itemName": target_item.name
        })

        # Place back the moved items
        step_counter += 1
        for item in reversed(items_to_move):
            step_counter += 1
            steps.append({
                "step": step_counter,
                "action": "placeBack",
                "itemId": item.item_id,
                "itemName": item.name
            })

        return steps

    def is_item_accessible(self, item: Item) -> bool:
        """Check if an item is directly accessible from container opening"""
        if item not in self.items:
            return False

        # Item is accessible if it's at depth 0
        x, y, z = item.position
        return y == 0

    def to_dict(self) -> Dict:
        """Convert container to dictionary representation"""
        return {
            "containerId": self.container_id,
            "zone": self.zone,
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "freeVolume": self.get_free_volume(),
            "itemCount": len(self.items)
        }


# API Request/Response Models
class ItemModel(BaseModel):
    itemId: str
    name: str
    width: int
    depth: int
    height: int
    priority: int
    expiryDate: Optional[str] = None
    usageLimit: Optional[int] = None
    preferredZone: Optional[str] = None
    mass: Optional[float] = 1.0


class ContainerModel(BaseModel):
    containerId: str
    zone: str
    width: int
    depth: int
    height: int


class PlacementRequest(BaseModel):
    items: List[ItemModel]
    containers: List[ContainerModel]


class RetrieveRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str


class PlaceRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str
    containerId: str
    position: Coordinates


class SimulateRequest(BaseModel):
    numOfDays: Optional[int] = None
    toTimestamp: Optional[str] = None
    itemsToBeUsedPerDay: List[Dict[str, str]]


class WasteReturnRequest(BaseModel):
    undockingContainerId: str
    undockingDate: str
    maxWeight: float


class UndockingRequest(BaseModel):
    undockingContainerId: str
    timestamp: str


# Optimization algorithms
def build_kdtree(containers: List[Container]):
    """Builds a KD-Tree for container lookup by dimensions"""
    positions = [(c.width, c.depth, c.height, c.get_free_volume()) for c in containers]
    return cKDTree(positions), positions


def find_best_container(item: Item, containers: List[Container], kdtree, kdtree_data, preferred_zone: str = None) -> \
Optional[Tuple[Container, Tuple[int, int, int]]]:
    """Find the best container for the item using KD-Tree and multithreading"""
    # Create query vector with item dimensions and some free space
    item_volume = item.width * item.depth * item.height
    query = (item.width, item.depth, item.height, item_volume * 1.5)  # Add some buffer

    # Query k-nearest containers by dimensions
    k = min(10, len(containers))
    if k == 0:
        return None

    distances, indices = kdtree.query([query], k=k)

    # First try preferred zone if specified
    preferred_containers = []
    other_containers = []

    for idx in indices[0]:
        container = containers[idx]
        if preferred_zone and container.zone == preferred_zone:
            preferred_containers.append(container)
        else:
            other_containers.append(container)

    # Try preferred containers first
    all_containers = preferred_containers + other_containers

    # Find position using multithreading
    def check_container(container):
        position = container.find_position(item)
        if position:
            return (container, position)
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(check_container, all_containers))

    # Filter out None results and return first valid result
    valid_results = [r for r in results if r is not None]
    if valid_results:
        return valid_results[0]

    return None


def rearrange_to_fit(new_items: List[Item], containers: List[Container]) -> Tuple[bool, List[Dict], List[Dict]]:
    """Place multiple items in containers with potential rearrangement"""
    # Sort items by priority (high to low)
    sorted_items = sorted(new_items, key=lambda x: x.priority, reverse=True)

    # Build KD-Tree once
    kdtree, kdtree_data = build_kdtree(containers)

    placements = []
    rearrangements = []
    step_counter = 1

    # First attempt: place high priority items without rearrangement
    unplaced_items = []
    for item in sorted_items:
        result = find_best_container(item, containers, kdtree, kdtree_data, item.preferred_zone)
        if result:
            container, position = result
            item.position = position
            container.add_item(item)

            placements.append({
                "itemId": item.item_id,
                "containerId": container.container_id,
                "position": {
                    "startCoordinates": {
                        "width": position[0],
                        "depth": position[1],
                        "height": position[2]
                    },
                    "endCoordinates": {
                        "width": position[0] + item.width,
                        "depth": position[1] + item.depth,
                        "height": position[2] + item.height
                    }
                }
            })
        else:
            unplaced_items.append(item)

    # If we have unplaced items, try simple rearrangement
    if unplaced_items:
        # Try to move low priority items to make space for high priority ones
        for unplaced_item in unplaced_items:
            # Find items with lower priority that could be moved
            low_priority_items = []
            for container in containers:
                for existing_item in container.items:
                    if existing_item.priority < unplaced_item.priority:
                        low_priority_items.append((existing_item, container))

            # Sort by priority (lowest first)
            low_priority_items.sort(key=lambda x: x[0].priority)

            # Try removing low priority items to make space
            for existing_item, existing_container in low_priority_items:
                # Remove the item temporarily
                existing_container.remove_item(existing_item)

                # See if our unplaced item fits now
                result = find_best_container(unplaced_item, containers, kdtree, kdtree_data,
                                             unplaced_item.preferred_zone)

                if result:
                    # It fits! Record the rearrangement
                    container, position = result

                    # Record the removal
                    rearrangements.append({
                        "step": step_counter,
                        "action": "remove",
                        "itemId": existing_item.item_id,
                        "fromContainer": existing_container.container_id,
                        "fromPosition": {
                            "startCoordinates": {
                                "width": existing_item.position[0],
                                "depth": existing_item.position[1],
                                "height": existing_item.position[2]
                            },
                            "endCoordinates": {
                                "width": existing_item.position[0] + existing_item.width,
                                "depth": existing_item.position[1] + existing_item.depth,
                                "height": existing_item.position[2] + existing_item.height
                            }
                        },
                        "toContainer": None,
                        "toPosition": None
                    })
                    step_counter += 1

                    # Place the unplaced item
                    unplaced_item.position = position
                    container.add_item(unplaced_item)

                    rearrangements.append({
                        "step": step_counter,
                        "action": "place",
                        "itemId": unplaced_item.item_id,
                        "fromContainer": None,
                        "fromPosition": None,
                        "toContainer": container.container_id,
                        "toPosition": {
                            "startCoordinates": {
                                "width": position[0],
                                "depth": position[1],
                                "height": position[2]
                            },
                            "endCoordinates": {
                                "width": position[0] + unplaced_item.width,
                                "depth": position[1] + unplaced_item.depth,
                                "height": position[2] + unplaced_item.height
                            }
                        }
                    })
                    step_counter += 1

                    placements.append({
                        "itemId": unplaced_item.item_id,
                        "containerId": container.container_id,
                        "position": {
                            "startCoordinates": {
                                "width": position[0],
                                "depth": position[1],
                                "height": position[2]
                            },
                            "endCoordinates": {
                                "width": position[0] + unplaced_item.width,
                                "depth": position[1] + unplaced_item.depth,
                                "height": position[2] + unplaced_item.height
                            }
                        }
                    })

                    # Try to find a new place for the removed item
                    new_result = find_best_container(existing_item, containers, kdtree, kdtree_data,
                                                     existing_item.preferred_zone)

                    if new_result:
                        new_container, new_position = new_result
                        existing_item.position = new_position
                        new_container.add_item(existing_item)

                        # Record the new placement
                        rearrangements.append({
                            "step": step_counter,
                            "action": "place",
                            "itemId": existing_item.item_id,
                            "fromContainer": None,
                            "fromPosition": None,
                            "toContainer": new_container.container_id,
                            "toPosition": {
                                "startCoordinates": {
                                    "width": new_position[0],
                                    "depth": new_position[1],
                                    "height": new_position[2]
                                },
                                "endCoordinates": {
                                    "width": new_position[0] + existing_item.width,
                                    "depth": new_position[1] + existing_item.depth,
                                    "height": new_position[2] + existing_item.height
                                }
                            }
                        })
                        step_counter += 1

                        # Successfully rearranged
                        unplaced_items.remove(unplaced_item)
                        break
                    else:
                        # Restore the item, this rearrangement doesn't work
                        existing_item.position = existing_item.position  # Keep original position
                        existing_container.add_item(existing_item)
                else:
                    # Restore the item, our unplaced item still doesn't fit
                    existing_container.add_item(existing_item)

    # Return success if all items were placed
    success = len(unplaced_items) == 0
    return success, placements, rearrangements


 def log_action(action_type: str, user_id: str, item_id: str, details: Dict = None):
     """Log an action in the system"""
     log_entry = {
         "timestamp": datetime.now().isoformat(),
         "userId": user_id,
         "actionType": action_type,
         "itemId": item_id,
        "details": details or {}
    }
    global_state.logs.append(log_entry)
    return log_entry


def get_waste_items() -> List[Dict]:
    """Identify waste items in the system"""
    waste_items = []
    for item_id, item in global_state.items.items():
        is_waste, reason = item.is_waste(global_state.current_date)
        if is_waste:
            container_id = item.container_id
            if container_id:
                container = global_state.containers.get(container_id)
                if container:
                    position = item.position
                    waste_items.append({
                        "itemId": item.item_id,
                        "name": item.name,
                        "reason": reason,
                        "containerId": container_id,
                        "position": {
                            "startCoordinates": {
                                "width": position[0],
                                "depth": position[1],
                                "height": position[2]
                            },
                            "endCoordinates": {
                                "width": position[0] + item.width,
                                "depth": position[1] + item.depth,
                                "height": position[2] + item.height
                            }
                        }
                    })
    return waste_items


# API Implementations
@app.post("/api/placement")
async def placement_api(request: PlacementRequest):
    """API to suggest placement for items"""
    start_time = time.time()

    # Convert request models to our internal models
    new_items = []
    for item_data in request.items:
        item = Item(
            item_id=item_data.itemId,
            name=item_data.name,
            width=item_data.width,
            depth=item_data.depth,
            height=item_data.height,
            priority=item_data.priority,
            expiry_date=item_data.expiryDate,
            usage_limit=item_data.usageLimit,
            preferred_zone=item_data.preferredZone,
            mass=item_data.mass if item_data.mass else 1.0
        )
        new_items.append(item)
        global_state.items[item.item_id] = item

    # Add containers if they don't exist
    containers = []
    for container_data in request.containers:
        container_id = container_data.containerId
        if container_id in global_state.containers:
            containers.append(global_state.containers[container_id])
        else:
            container = Container(
                container_id=container_id,
                width=container_data.width,
                depth=container_data.depth,
                height=container_data.height,
                zone=container_data.zone
            )
            containers.append(container)
            global_state.containers[container_id] = container

    # Update global KD-Tree
    global_state.kdtree, global_state.kdtree_data = build_kdtree(list(global_state.containers.values()))

    # Find placements with potential rearrangements
    success, placements, rearrangements = rearrange_to_fit(new_items, containers)

    # Log the placement action
    for placement in placements:
        log_action("placement", "system", placement["itemId"], {"containerId": placement["containerId"]})

    execution_time = time.time() - start_time
    print(f"Placement API execution time: {execution_time:.4f} seconds")

    return {
        "success": success,
        "placements": placements,
        "rearrangements": rearrangements,
        "executionTime": execution_time
    }


@app.get("/api/search")
async def search_api(itemId: str = None, itemName: str = None, userId: str = None):
    """API to search for an item"""
    start_time = time.time()

    item = None
    if itemId and itemId in global_state.items:
        item = global_state.items[itemId]
    elif itemName:
        for i in global_state.items.values():
            if i.name == itemName:
                item = i
                break

    if not item:
        return {"success": True, "found": False}

    container_id = item.container_id
    if not container_id or container_id not in global_state.containers:
        return {"success": True, "found": False}

    container = global_state.containers[container_id]
    position = item.position
    retrieval_steps = container.get_retrieval_steps(item)

    log_action("search", userId or "anonymous", item.item_id, {"containerId": container_id})

    execution_time = time.time() - start_time
    print(f"Search API execution time: {execution_time:.4f} seconds")

    return {
        "success": True,
        "found": True,
        "item": {
            "itemId": item.item_id,
            "name": item.name,
            "containerId": container_id,
            "zone": container.zone,
            "position": {
                "startCoordinates": {
                    "width": position[0],
                    "depth": position[1],
                    "height": position[2]
                },
                "endCoordinates": {
                    "width": position[0] + item.width,
                    "depth": position[1] + item.depth,
                    "height": position[2] + item.height
                }
            }
        },
        "retrievalSteps": retrieval_steps,
        "executionTime": execution_time
    }


@app.post("/api/retrieve")
async def retrieve_api(request: RetrieveRequest):
    """API to mark an item as retrieved/used"""
    item_id = request.itemId
    if item_id not in global_state.items:
        return {"success": False}

    item = global_state.items[item_id]
    container_id = item.container_id

    if container_id and container_id in global_state.containers:
        container = global_state.containers[container_id]

        # Check if it's retrievable
        if not container.is_item_accessible(item):
            # Would need to move other items first
            return {"success": False, "message": "Item not directly accessible"}

        # Update usage counter
        if item.remaining_uses is not None:
            item.remaining_uses -= 1

        # Log the retrieval
        timestamp = datetime.fromisoformat(request.timestamp)
        log_action("retrieval", request.userId, item_id, {
            "fromContainer": container_id,
            "timestamp": timestamp.isoformat()
        })

        # Remove item from container if it's now waste
        is_waste, reason = item.is_waste(global_state.current_date)
        if is_waste:
            container.remove_item(item)
            log_action("waste", "system", item_id, {"reason": reason})

        return {"success": True}

    return {"success": False}


@app.post("/api/place")
async def place_api(request: PlaceRequest):
    """API to place an item in a container"""
    item_id = request.itemId
    container_id = request.containerId
    position = (
        request.position.startCoordinates.width,
        request.position.startCoordinates.depth,
        request.position.startCoordinates.height
    )

    if item_id not in global_state.items or container_id not in global_state.containers:
        return {"success": False}

    item = global_state.items[item_id]
    container = global_state.containers[container_id]

    # Check if position is valid
    if not container.can_fit_at_position(item, position):
        return {"success": False, "message": "Invalid position"}

    # Remove from current container if any
    if item.container_id:
        old_container = global_state.containers.get(item.container_id)
        if old_container:
            old_container.remove_item(item)

    # Place in new container
    item.position = position
    container.add_item(item)

    # Log the placement
    timestamp = datetime.fromisoformat(request.timestamp)
    log_action("placement", request.userId, item_id, {
        "toContainer": container_id,
        "timestamp": timestamp.isoformat()
    })

    return {"success": True}


@app.get("/api/waste/identify")
async def identify_waste_api():
    """API to identify waste items"""
    waste_items = get_waste_items()
    return {"success": True, "wasteItems": waste_items}


@app.post("/api/waste/return-plan")
async def return_plan_api(request: WasteReturnRequest):
    """API to create a waste return plan"""
    undocking_container_id = request.undockingContainerId
    max_weight = request.maxWeight
    undocking_date = request.undockingDate

    if undocking_container_id not in global_state.containers:
        return {"success": False, "message": "Undocking container not found"}

    undocking_container = global_state.containers[undocking_container_id]
    waste_items = get_waste_items()

    # Sort waste items by priority (lower priority waste gets removed first)
    # This ensures high-priority waste is handled first if we exceed weight limits
    sorted_waste_items = sorted(waste_items, key=lambda x: global_state.items[x["itemId"]].priority)

    return_plan = []
    retrieval_steps = []
    return_items = []
    total_volume = 0
    total_weight = 0
    step_counter = 1

    # Process each waste item
    for waste_item in sorted_waste_items:
        item_id = waste_item["itemId"]
        item = global_state.items[item_id]

        # Check if adding this item would exceed max weight
        if total_weight + item.mass > max_weight:
            continue

        # Get retrieval steps for this waste item
        item_retrieval_steps = calculate_retrieval_steps(item_id)

        # Update the counter for each step
        for step in item_retrieval_steps:
            step["step"] = step_counter
            step_counter += 1
            retrieval_steps.append(step)

        # Add item to return plan
        return_plan.append({
            "step": step_counter,
            "itemId": item_id,
            "itemName": item.name,
            "fromContainer": waste_item["containerId"],
            "toContainer": undocking_container_id
        })
        step_counter += 1

        # Update total volume and weight
        item_volume = item.width * item.depth * item.height
        total_volume += item_volume
        total_weight += item.mass

        # Add to return items list
        return_items.append({
            "itemId": item_id,
            "name": item.name,
            "reason": waste_item["reason"]
        })

    # Create the return manifest
    return_manifest = {
        "undockingContainerId": undocking_container_id,
        "undockingDate": undocking_date,
        "returnItems": return_items,
        "totalVolume": total_volume,
        "totalWeight": total_weight
    }

    return {
        "success": True,
        "returnPlan": return_plan,
        "retrievalSteps": retrieval_steps,
        "returnManifest": return_manifest
    }


def calculate_retrieval_steps(item_id):
    """Calculate steps required to retrieve an item"""
    steps = []
    item = global_state.items[item_id]
    container_id = item.container_id

    if not container_id:
        return steps

    container = global_state.containers[container_id]
    item_position = item.position

    # Find items that need to be removed to access the target item
    blocking_items = find_blocking_items(container_id, item_position)
    step_count = 1

    # Add steps to remove blocking items
    for blocking_item_id in blocking_items:
        blocking_item = global_state.items[blocking_item_id]
        steps.append({
            "step": step_count,
            "action": "remove",
            "itemId": blocking_item_id,
            "itemName": blocking_item.name
        })
        step_count += 1

        steps.append({
            "step": step_count,
            "action": "setAside",
            "itemId": blocking_item_id,
            "itemName": blocking_item.name
        })
        step_count += 1

    # Add step to retrieve the target item
    steps.append({
        "step": step_count,
        "action": "retrieve",
        "itemId": item_id,
        "itemName": item.name
    })
    step_count += 1

    # Add steps to place back all removed items
    for blocking_item_id in reversed(blocking_items):
        blocking_item = global_state.items[blocking_item_id]
        steps.append({
            "step": step_count,
            "action": "placeBack",
            "itemId": blocking_item_id,
            "itemName": blocking_item.name
        })
        step_count += 1

    return steps


def find_blocking_items(container_id, target_position):
    """Find items blocking access to the target position"""
    container = global_state.containers[container_id]
    blocking_items = []

    # Get items in the container
    container_items = [item_id for item_id, item in global_state.items.items()
                       if item.container_id == container_id]

    # Calculate which items block the target item's path to the open face
    for item_id in container_items:
        item = global_state.items[item_id]
        if item_id == target_position["itemId"]:
            continue

        if is_blocking(item.position, target_position):
            blocking_items.append(item_id)

    return blocking_items


def is_blocking(item1_pos, item2_pos):
    """Determine if item1 is blocking access to item2"""
    # Item1 blocks item2 if:
    # 1. Item1 is between item2 and the open face (depth-wise)
    # 2. Item1 overlaps with item2 in both width and height dimensions

    # Check if item1 is closer to open face than item2
    if item1_pos["startCoordinates"]["depth"] < item2_pos["startCoordinates"]["depth"]:
        # Check if there's width overlap
        width_overlap = (
                item1_pos["startCoordinates"]["width"] < item2_pos["endCoordinates"]["width"] and
                item1_pos["endCoordinates"]["width"] > item2_pos["startCoordinates"]["width"]
        )

        # Check if there's height overlap
        height_overlap = (
                item1_pos["startCoordinates"]["height"] < item2_pos["endCoordinates"]["height"] and
                item1_pos["endCoordinates"]["height"] > item2_pos["startCoordinates"]["height"]
        )

        return width_overlap and height_overlap

    return False


def get_waste_items():
    """Get all waste items in the system"""
    waste_items = []
    current_time = datetime.now().isoformat()

    for item_id, item in global_state.items.items():
        # Skip items already in the undocking container
        if item.container_id == request.undockingContainerId:
            continue

        # Check if item is expired
        if item.expiry_date and item.expiry_date < current_time:
            waste_items.append({
                "itemId": item_id,
                "name": item.name,
                "reason": "Expired",
                "containerId": item.container_id,
                "position": item.position
            })
        # Check if item is out of uses
        elif item.usage_limit is not None and item.usage_limit <= 0:
            waste_items.append({
                "itemId": item_id,
                "name": item.name,
                "reason": "Out of Uses",
                "containerId": item.container_id,
                "position": item.position
            })

    return waste_items


@app.post("/api/waste/complete-undocking")
async def complete_undocking_api(request: UndockingRequest):
    """
    API to complete the undocking process for waste items

    This endpoint:
    1. Removes all waste items from the specified undocking container
    2. Updates the system state to reflect the undocking
    3. Returns the number of items removed
    """
    undocking_container_id = request.undockingContainerId
    timestamp = request.timestamp

    # Validate that the container exists
    if undocking_container_id not in global_state.containers:
        return {"success": False, "message": "Undocking container not found"}

    # Get all items in the undocking container
    items_to_remove = []
    for item_id, item in global_state.items.items():
        if item.container_id == undocking_container_id:
            items_to_remove.append(item_id)

    # Track how many items are removed
    removed_count = 0

    # Remove items from the system
    for item_id in items_to_remove:
        # Log the disposal action
        log_entry = {
            "timestamp": timestamp,
            "userId": "system",  # System action for undocking
            "actionType": "disposal",
            "itemId": item_id,
            "details": {
                "fromContainer": undocking_container_id,
                "toContainer": "undocked",
                "reason": "Waste Return"
            }
        }
        global_state.logs.append(log_entry)

        # Remove the item from the system
        del global_state.items[item_id]
        removed_count += 1

    # Return the success response with the count of removed items
    return {
        "success": True,
        "itemsRemoved": removed_count
    }


@app.post("/api/retrieve")
async def retrieve_item(retrieval: dict):
    # Extract data
    item_id = retrieval["itemId"]
    user_id = retrieval["userId"]

    if item_id not in items_db:
        return {"success": False}

    # Update the item's usage count
    items_db[item_id]["remainingUses"] -= 1

    container_id = item_placements.get(item_id, {}).get("containerId", "unknown")

    # Log the action
    log_action(
        user_id=user_id,
        action_type="retrieval",
        item_id=item_id,
        details={
            "fromContainer": container_id
        }
    )

    return {"success": True}


@app.post("/api/place")
async def place_item(item_placement: dict):
    # Extract data
    item_id = item_placement["itemId"]
    user_id = item_placement["userId"]
    container_id = item_placement["containerId"]
    position = item_placement["position"]

    # Store the placement
    item_placements[item_id] = {
        "containerId": container_id,
        "position": position
    }

    # Log the action
    log_action(
        user_id=user_id,
        action_type="placement",
        item_id=item_id,
        details={
            "toContainer": container_id
        }
    )

    return {"success": True}


@app.get("/api/logs")
async def get_logs(
        startDate: str = Query(...),
        endDate: str = Query(...),
        itemId: Optional[str] = None,
        userId: Optional[str] = None,
        actionType: Optional[str] = None
):
    try:
        start_date = datetime.fromisoformat(startDate.replace("Z", "+00:00"))
        end_date = datetime.fromisoformat(endDate.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    # Filter logs based on parameters
    filtered_logs = []
    for log in logs_db:
        log_date = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))

        # Date range check
        if not (start_date <= log_date <= end_date):
            continue

        # Item ID check
        if itemId and log["itemId"] != itemId:
            continue

        # User ID check
        if userId and log["userId"] != userId:
            continue

        # Action type check
        if actionType and log["actionType"] != actionType:
            continue

        # If passed all filters, add to result
        filtered_logs.append(log)

    return {"logs": filtered_logs}


@app.get("/api/export/arrangement")
async def export_arrangement():
    # Create a temporary CSV file
    file_path = "temp_arrangement.csv"

    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Item ID', 'Container ID', 'Coordinates (W1,D1,H1),(W2,D2,H2)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item_id, placement in item_placements.items():
            pos = placement["position"]
            coord_str = f"({pos['startCoordinates']['width']},{pos['startCoordinates']['depth']},{pos['startCoordinates']['height']}),({pos['endCoordinates']['width']},{pos['endCoordinates']['depth']},{pos['endCoordinates']['height']})"

            writer.writerow({
                'Item ID': item_id,
                'Container ID': placement["containerId"],
                'Coordinates (W1,D1,H1),(W2,D2,H2)': coord_str
            })

    # Return the file and then delete it
    response = FileResponse(file_path, filename="arrangement.csv")

    # Delete the file after sending (schedule for deletion)
    # In a real app, this would need proper cleanup handling
    return response


@app.post("/api/import/containers")
async def import_containers(file: UploadFile = File(...)):
    containers_imported = 0
    errors = []

    # Read the CSV file
    content = await file.read()
    csv_reader = csv.DictReader(io.StringIO(content.decode('utf-8')))

    for row_num, row in enumerate(csv_reader, start=2):  # Start from 2 to account for header row
        try:
            # Convert row data to proper types
            container_id = row.get("Container ID", "").strip()
            if not container_id:
                errors.append({"row": row_num, "message": "Missing Container ID"})
                continue

            # Process the row data
            container_data = {
                "containerId": container_id,
                "zone": row.get("Zone", "").strip(),
                "width": float(row.get("Width(cm)", 0)),
                "depth": float(row.get("Depth(cm)", 0)),
                "height": float(row.get("Height(height)", 0))
            }

            # Validate data
            if container_data["width"] <= 0 or container_data["depth"] <= 0 or container_data["height"] <= 0:
                errors.append({"row": row_num, "message": "Invalid dimensions"})
                continue

            # Store the container
            containers_db[container_id] = container_data
            containers_imported += 1

        except Exception as e:
            errors.append({"row": row_num, "message": f"Error processing row: {str(e)}"})

    return {
        "success": len(errors) == 0,
        "containersImported": containers_imported,
        "errors": errors
    }


@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    items_imported = 0
    errors = []

    # Read the CSV file
    content = await file.read()
    csv_reader = csv.DictReader(io.StringIO(content.decode('utf-8')))

    for row_num, row in enumerate(csv_reader, start=2):  # Start from 2 to account for header row
        try:
            # Convert row data to proper types
            item_id = row.get("Item ID", "").strip()
            if not item_id:
                errors.append({"row": row_num, "message": "Missing Item ID"})
                continue

            # Process the row data
            item_data = {
                "itemId": item_id,
                "name": row.get("Name", "").strip(),
                "width": float(row.get("Width (cm)", 0)),
                "depth": float(row.get("Depth (cm)", 0)),
                "height": float(row.get("Height (cm)", 0)),
                "priority": int(row.get("Priority (1-100)", 0)),
                "expiryDate": row.get("Expiry Date (ISO Format)", "N/A").strip(),
                "usageLimit": int(row.get("Usage Limit", 0)),
                "remainingUses": int(row.get("Usage Limit", 0)),  # Initialize remaining uses
                "preferredZone": row.get("Preferred Zone", "").strip()
            }

            # Validate data
            if item_data["width"] <= 0 or item_data["depth"] <= 0 or item_data["height"] <= 0:
                errors.append({"row": row_num, "message": "Invalid dimensions"})
                continue

            if not (0 <= item_data["priority"] <= 100):
                errors.append({"row": row_num, "message": "Priority must be between 0 and 100"})
                continue

            # Store the item
            items_db[item_id] = item_data
            items_imported += 1

        except Exception as e:
            errors.append({"row": row_num, "message": f"Error processing row: {str(e)}"})

    return {
        "success": len(errors) == 0,
        "itemsImported": items_imported,
        "errors": errors
    }

