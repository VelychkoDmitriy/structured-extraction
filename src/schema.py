"""Extraction schema: raw resume text -> structured Resume.

Field names follow the KSE-RESEARCH-Group/Work_UA_resumes dataset so results can be compared with it.
Changes vs the dataset: no service fields (id, creation_date, other_resumes), no sensitive fields
(veteran, disability), no industry / additional_info; levels and employment type are enums;
dates are "YYYY-MM" or "YYYY"; missing_fields lists what the text does not contain.
Labelling rules: LABELING_GUIDE.md.
"""
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

DATE_PATTERN = r"^\d{4}(-\d{2})?$"


class LanguageLevel(str, Enum):
    """Work.ua scale."""
    native = "рідна"
    fluent = "вільно"
    advanced = "просунутий"
    upper_intermediate = "вище середнього"
    intermediate = "середній"
    basic = "початковий"
    unknown = "unknown"


class EducationLevel(str, Enum):
    higher = "вища"
    incomplete_higher = "незакінчена вища"
    vocational = "середня спеціальна"
    secondary = "середня"
    unknown = "unknown"


class EmploymentType(str, Enum):
    full_time = "повна"
    part_time = "неповна"


MissingField = Literal[
    "title", "age", "city", "desired_salary", "employment_type", "work_location_preference",
    "driver_license", "work_experiences", "educations", "additional_educations", "languages", "skills",
]


class WorkExperience(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: Optional[str] = Field(description="Job title as written")
    company: Optional[str] = Field(description="Employer as written; null if not named")
    city: Optional[str] = Field(description="City of this job if stated")
    start_date: Optional[str] = Field(pattern=DATE_PATTERN, description="YYYY-MM, or YYYY if only the year is known")
    end_date: Optional[str] = Field(pattern=DATE_PATTERN, description="YYYY-MM or YYYY; null if current or not stated")
    is_current: Optional[bool] = Field(description="true for 'по нині'/'present'; null if the text does not say")
    responsibilities: Optional[str] = Field(default=None, description="Duties as written; extracted but not scored")


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")
    institution: Optional[str]
    faculty: Optional[str] = Field(description="Faculty or speciality as written")
    city: Optional[str]
    level: EducationLevel
    start_year: Optional[int] = Field(ge=1950, le=2030)
    end_year: Optional[int] = Field(ge=1950, le=2035)


class Language(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str = Field(description="Ukrainian name in lower case: 'англійська', 'українська', 'російська'…")
    level: LanguageLevel


class Resume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(description="Desired position: the resume headline")
    age: Optional[int] = Field(ge=14, le=90, description="Age as stated; if only a birth date is given, full years on the resume date")
    city: Optional[str] = Field(description="City of residence without prefixes like 'м.' or 'с.'")
    desired_salary: Optional[int] = Field(gt=0, description="Monthly salary in UAH; null for hourly or other currencies")
    employment_type: List[EmploymentType] = Field(description="Only if stated: 'повна' and/or 'неповна' зайнятість")
    work_location_preference: List[str] = Field(description="'Дистанційно' and/or cities the candidate is ready to work in")
    driver_license: Optional[bool] = Field(description="true if a licence is mentioned; null if not mentioned")
    work_experiences: List[WorkExperience] = Field(description="Jobs with a position or dates; study projects excluded")
    educations: List[Education]
    additional_educations: List[str] = Field(description="Courses and trainings, name as written")
    languages: List[Language]
    skills: List[str] = Field(description="Skills, tools and listed personal qualities, as written")
    missing_fields: List[MissingField] = Field(description="Top-level fields the text gives no value for")
