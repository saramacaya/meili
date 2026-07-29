-- Meili MVP Database Schema
-- This schema stores anonymous route-choice behaviour for the Meili prototype.

create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  user_type text not null,
  created_at timestamp with time zone default now()
);

create table if not exists routes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id) on delete cascade,
  origin_text text not null,
  destination_text text not null,
  initial_preference text not null,
  route_type text not null,
  estimated_time_minutes integer not null,
  distance_meters integer not null,
  safety_score integer not null,
  route_geometry_json jsonb,
  explanation text,
  created_at timestamp with time zone default now(),

  constraint valid_initial_preference
    check (initial_preference in ('fastest', 'balanced', 'safest')),

  constraint valid_route_type
    check (route_type in ('fastest', 'balanced', 'safest')),

  constraint valid_safety_score
    check (safety_score >= 0 and safety_score <= 100)
);

create table if not exists route_choices (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id) on delete cascade,
  chosen_route_id uuid references routes(id) on delete set null,
  fastest_route_id uuid references routes(id) on delete set null,
  balanced_route_id uuid references routes(id) on delete set null,
  safest_route_id uuid references routes(id) on delete set null,
  initial_preference text not null,
  final_choice_type text not null,
  extra_time_minutes integer,
  safety_gain integer,
  framing_group text,
  chosen_at timestamp with time zone default now(),

  constraint valid_choice_initial_preference
    check (initial_preference in ('fastest', 'balanced', 'safest')),

  constraint valid_final_choice_type
    check (final_choice_type in ('fastest', 'balanced', 'safest'))
);

create table if not exists feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id) on delete cascade,
  route_id uuid references routes(id) on delete set null,
  perceived_safety_rating integer not null,
  would_choose_again boolean not null,
  comment text,
  created_at timestamp with time zone default now(),

  constraint valid_perceived_safety_rating
    check (perceived_safety_rating >= 1 and perceived_safety_rating <= 5)
);