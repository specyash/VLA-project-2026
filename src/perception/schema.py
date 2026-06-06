"""Stable detection schema placeholder. YOLO converts the detected object into a schema class which will be used furthur int the codebase as the object."""

from dataclasses import dataclass

@dataclass
class DetectedObject:

    """image pixels here, later can be converted to table coordinates"""
    
    id:str
    base_color:str
    marked:bool
    label:str
    box:tuple[int,int,int,int] #x,y,width,height
    center_px:tuple[int,int] #x,y
    confidence:float
    area:float