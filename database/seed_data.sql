-- Meili MVP Seed Data
-- This inserts one fake test journey so we can understand how the tables connect.

do $$
declare
  test_user_id uuid;
  fastest_route_id uuid;
  balanced_route_id uuid;
  safest_route_id uuid;
begin
  -- 1. Create one anonymous test user/session
  insert into users (user_type)
  values ('commuter')
  returning id into test_user_id;

  -- 2. Create the Fastest route shown to this user
  insert into routes (
    user_id,
    origin_text,
    destination_text,
    initial_preference,
    route_type,
    estimated_time_minutes,
    distance_meters,
    safety_score,
    route_geometry_json,
    explanation
  )
  values (
    test_user_id,
    'Ruzafa',
    'Plaça de la Reina',
    'safest',
    'fastest',
    13,
    1300,
    60,
    null,
    'Shortest walking path through side streets, but with a lower safety score.'
  )
  returning id into fastest_route_id;

  -- 3. Create the Balanced route shown to this user
  insert into routes (
    user_id,
    origin_text,
    destination_text,
    initial_preference,
    route_type,
    estimated_time_minutes,
    distance_meters,
    safety_score,
    route_geometry_json,
    explanation
  )
  values (
    test_user_id,
    'Ruzafa',
    'Plaça de la Reina',
    'safest',
    'balanced',
    16,
    1400,
    74,
    null,
    'Mixes lit avenues with shorter connectors to balance time and perceived safety.'
  )
  returning id into balanced_route_id;

  -- 4. Create the Safest route shown to this user
  insert into routes (
    user_id,
    origin_text,
    destination_text,
    initial_preference,
    route_type,
    estimated_time_minutes,
    distance_meters,
    safety_score,
    route_geometry_json,
    explanation
  )
  values (
    test_user_id,
    'Ruzafa',
    'Plaça de la Reina',
    'safest',
    'safest',
    19,
    1600,
    85,
    null,
    'Routes along better-lit main streets and busier public areas.'
  )
  returning id into safest_route_id;

  -- 5. Save the final route choice
  -- In this example, the user initially preferred safest but finally chose balanced.
  insert into route_choices (
    user_id,
    chosen_route_id,
    fastest_route_id,
    balanced_route_id,
    safest_route_id,
    initial_preference,
    final_choice_type,
    extra_time_minutes,
    safety_gain,
    framing_group
  )
  values (
    test_user_id,
    balanced_route_id,
    fastest_route_id,
    balanced_route_id,
    safest_route_id,
    'safest',
    'balanced',
    3,
    14,
    'default'
  );

  -- 6. Save feedback for the route the user actually chose
  insert into feedback (
    user_id,
    route_id,
    perceived_safety_rating,
    would_choose_again,
    comment
  )
  values (
    test_user_id,
    balanced_route_id,
    4,
    true,
    'I chose the balanced route because it seemed safer than the fastest route without adding too much time.'
  );
end $$;
