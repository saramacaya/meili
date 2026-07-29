# Meili Technical Specification

## 1. Project summary

Meili is a behavioural decision system for urban navigation.

The original idea was a safety-focused navigation app that helps users move through cities with more confidence. The current MVP narrows that idea into a stronger applied analytics project: a prototype that shows users different route options, records how they compare travel time and perceived safety, and analyses their final decisions.

The MVP does not try to build a perfect commercial navigation app. Instead, it focuses on a clear behavioural question:

How do users trade off travel time and perceived safety when choosing a route?

## 2. Core product idea

The app shows users three route options:

- Fastest
- Balanced
- Safest

The user first states an initial preference before seeing the options. Then the app shows all three routes. After seeing the time and safety trade-offs, the user selects one final route and provides feedback.

The key behavioural comparison is:

initial preference vs final choice

Example:

Initial preference: fastest  
Final choice: safest

This means the user changed their mind after seeing the route information.

## 3. MVP user flow

The MVP flow is:

Home
→ Initial Preference
→ Route Comparison
→ Feedback

### Home

The user enters:

- Origin
- Destination
- User type

Example user types:

- Solo traveller
- Student or new resident
- Frequent traveller
- Local user
- Other

### Initial Preference

The user answers:

What matters most for this journey?

Options:

- Fastest
- Balanced
- Safest

Important rule:

The initial preference does not hide any route options. It only determines which option may be highlighted first.

### Route Comparison

The app shows all three route options as cards:

- Fastest Route
- Balanced Route
- Safest Route

Each route card should show:

- Estimated time
- Distance
- Safety score
- Short explanation
- Trade-off summary
- Choose this route button

For the MVP, the app should show only the currently selected route on the map. It should not show all three route lines on the map at once.

### Feedback

After selecting a final route, the user gives feedback.

The feedback screen collects:

- Perceived safety rating
- Whether the user would choose the route again
- Optional comment

## 4. Current Lovable prototype

Meili already has an early Lovable prototype with:

- Valencia map
- Route display
- Safety score
- Route summary card
- Visual identity and branding

This existing Lovable prototype will not be discarded. It will be reused as the visual foundation and refactored into the MVP behavioural route-choice flow.

The goal is to keep the current visual style while restructuring the product into:

Home → Initial Preference → Route Comparison → Feedback

## 5. MVP scope

### Build now

The MVP should include:

- Clickable Lovable prototype
- Origin and destination inputs
- Initial preference selection
- Three route comparison cards
- One selected route shown on the map
- Final route choice selection
- Feedback collection
- Backend route generation
- Supabase database
- Route choice logging
- Safety scoring model
- Basic routing logic
- Behavioural analytics notebook
- GitHub documentation
- Final report

### Postpone for later

The following features should not be built yet:

- Panic button
- Payment system
- Full login/authentication
- App Store launch
- Taxi integration
- Government dashboard
- Real-time crime API
- Full city routing
- AI recommendation engine
- Social/community feed
- Complex emergency features

Scope rule:

If a feature does not directly support behavioural route-choice analysis, safety-aware route comparison, or applied analytics, it should be postponed.

## 6. Technology stack

### Frontend

Tool: Lovable

Purpose:

- Build mobile app screens
- Create the user flow
- Display route cards
- Display selected route on map
- Send route requests to the backend
- Collect feedback

Lovable should be used as the interface layer, not the main logic layer.

### Backend

Tool: Python with FastAPI

Purpose:

- Receive route requests
- Generate three route options
- Calculate or return safety scores
- Save route data
- Save final route choices
- Save feedback
- Connect the frontend to the database

### Database

Tool: Supabase/PostgreSQL

Purpose:

- Store anonymous users
- Store route options
- Store final route choices
- Store user feedback
- Later, store street segments and incident data

### Analytics

Tool: Python notebooks

Libraries:

- pandas
- numpy
- matplotlib
- scikit-learn

Purpose:

- Analyse route choices
- Study time/safety trade-offs
- Calculate how often users change from initial preference to final choice
- Create charts for the final report
- Run simple behavioural models

### Maps

Initial approach:

- Use the existing Lovable map display if possible
- Keep routing simple at first
- Show only the selected route on the map

Later options:

- Mapbox
- OpenStreetMap
- More realistic routing API

## 7. Database design

The MVP uses four main tables.

### users

Purpose:

Store anonymous user profile type.

Columns:

- id
- user_type
- created_at

Example user_type values:

- solo_traveller
- student_new_resident
- frequent_traveller
- local_user
- other

### routes

Purpose:

Store route options shown to the user.

Columns:

- id
- user_id
- origin_text
- destination_text
- initial_preference
- route_type
- estimated_time_minutes
- distance_meters
- safety_score
- route_geometry_json
- explanation
- created_at

Allowed route_type values:

- fastest
- balanced
- safest

### route_choices

Purpose:

Store the user's final route choice.

This is the most important table for behavioural analytics.

Columns:

- id
- user_id
- chosen_route_id
- fastest_route_id
- balanced_route_id
- safest_route_id
- initial_preference
- final_choice_type
- extra_time_minutes
- safety_gain
- framing_group
- chosen_at

Important comparison:

initial_preference vs final_choice_type

### feedback

Purpose:

Store user perception after choosing a route.

Columns:

- id
- user_id
- route_id
- perceived_safety_rating
- would_choose_again
- comment
- created_at

## 8. API endpoints

### GET /health

Purpose:

Check whether the backend is running.

Example response:

```json
{
  "status": "Meili backend is running"
}